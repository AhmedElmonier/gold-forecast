import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class GoldForecastModel:
    def __init__(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0):
        """
        Initializes the Prophet model for Gold price forecasting.
        """
        self.model = Prophet(
            daily_seasonality=False,   # Gold markets are closed on weekends, daily seasonality might be noisy
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        self.model.add_regressor('USD_Index')
        self.model.add_regressor('Treasury_Yield')
        self.model.add_regressor('SP500')
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """
        Fits the Prophet model to the historical data.
        
        Args:
            df (pd.DataFrame): Dataframe with 'ds' and 'y' columns.
        """
        logger.info("Fitting Prophet model on historical data...")
        self.model.fit(df)
        self.is_fitted = True
        logger.info("Model fitting complete.")
        
    def predict(self, historical_df: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
        """
        Generates predictions for future dates, using the last known values for regressors.
        
        Args:
            historical_df (pd.DataFrame): The dataframe containing historical regressors.
            days_ahead (int): Number of days to forecast into the future.
            
        Returns:
            pd.DataFrame: Forecast dataframe containing 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
            
        logger.info(f"Generating forecast for {days_ahead} days ahead...")
        future = self.model.make_future_dataframe(periods=days_ahead)
        
        # For our 30-day forecast, we will naively assume the macro variables 
        # remain exactly what they were on the last available day of historical data.
        last_usd = historical_df.iloc[-1]['USD_Index']
        last_tnx = historical_df.iloc[-1]['Treasury_Yield']
        last_sp500 = historical_df.iloc[-1]['SP500']
        
        future['USD_Index'] = last_usd
        future['Treasury_Yield'] = last_tnx
        future['SP500'] = last_sp500
        
        # Override the historical part of the future dataframe with actual historical regressor values
        # so the model fits the past correctly when plotting
        future_merged = pd.merge(future, historical_df[['ds', 'USD_Index', 'Treasury_Yield', 'SP500']], on='ds', how='left')
        future['USD_Index'] = future_merged['USD_Index_y'].fillna(last_usd)
        future['Treasury_Yield'] = future_merged['Treasury_Yield_y'].fillna(last_tnx)
        future['SP500'] = future_merged['SP500_y'].fillna(last_sp500)

        forecast = self.model.predict(future)
        return forecast

    def evaluate(self, initial: str = '1095 days', period: str = '180 days', horizon: str = '30 days') -> pd.DataFrame:
        """
        Performs Time-Series Cross-Validation.
        
        Args:
            initial (str): Initial training period.
            period (str): Spacing between cutoff dates.
            horizon (str): Forecast horizon.
            
        Returns:
            pd.DataFrame: Cross-validation performance metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluating.")
            
        logger.info(f"Running cross-validation with initial={initial}, period={period}, horizon={horizon}...")
        df_cv = cross_validation(self.model, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv)
        
        # Log the average MAPE for the horizon
        avg_mape = df_p['mape'].mean()
        logger.info(f"Cross-validation complete. Average MAPE: {avg_mape:.2%}")
        
        return df_p

class XGBoostForecaster:
    def __init__(self):
        """
        Initializes the XGBoost model to predict n-days ahead using lagged features.
        """
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.is_fitted = False
        # The technical indicators and macro regressors to use as features
        self.features = ['SMA_20', 'SMA_50', 'RSI_14', 'USD_Index', 'Treasury_Yield', 'SP500']
        
    def fit(self, df: pd.DataFrame, days_ahead: int = 30):
        """
        Fits the XGBoost model to predict the price `days_ahead` into the future.
        """
        logger.info(f"Fitting XGBoost model to predict {days_ahead} days ahead...")
        train_df = df.copy()
        # Create target variable: the price 'days_ahead' in the future
        train_df['target'] = train_df['y'].shift(-days_ahead)
        train_df.dropna(inplace=True)
        
        X = train_df[self.features]
        y = train_df['target']
        
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("XGBoost model fitting complete.")
        
    def predict_current(self, df: pd.DataFrame) -> float:
        """
        Predicts the future price based on the very latest available data row.
        """
        if not self.is_fitted:
            raise ValueError("XGBoost model must be fitted before predicting.")
            
        latest_features = df.iloc[-1:][self.features]
        pred = self.model.predict(latest_features)
        return float(pred[0])

def generate_insights(forecast: pd.DataFrame, historical_data: pd.DataFrame, days_ahead: int, xgb_prediction: float = None) -> dict:
    """
    Generates actionable insights based on the forecast and current technical indicators.
    Ensembles Prophet and XGBoost predictions if both are available.
    
    Args:
        forecast (pd.DataFrame): Forecast dataframe from Prophet.
        historical_data (pd.DataFrame): The preprocessed historical data with indicators.
        days_ahead (int): The forecast horizon in days.
        xgb_prediction (float): The prediction from the XGBoost model.
        
    Returns:
        dict: A dictionary containing forecasted values, a trend insight, and explicit trading signals.
    """
    latest_hist = historical_data.iloc[-1]
    current_price = latest_hist['y']
    sma_20 = latest_hist['SMA_20']
    sma_50 = latest_hist['SMA_50']
    rsi_14 = latest_hist['RSI_14']
    
    # Get the row corresponding to the final predicted day
    future_pred = forecast.iloc[-1]
    prophet_price = future_pred['yhat']
    lower_bound = future_pred['yhat_lower']
    upper_bound = future_pred['yhat_upper']
    
    # Ensemble Logic: Average Prophet and XGBoost if XGBoost is provided
    if xgb_prediction is not None:
        ensemble_price = (prophet_price + xgb_prediction) / 2
        predicted_price = ensemble_price
        logger.info(f"Ensemble Forecast: Prophet=${prophet_price:.2f}, XGBoost=${xgb_prediction:.2f} -> Average=${predicted_price:.2f}")
    else:
        predicted_price = prophet_price
    
    price_diff = predicted_price - current_price
    pct_change = (price_diff / current_price) * 100
    
    if pct_change > 1.5:
        forecast_trend = "Strong Bullish 📈"
    elif pct_change > 0:
        forecast_trend = "Slightly Bullish ↗️"
    elif pct_change < -1.5:
        forecast_trend = "Strong Bearish 📉"
    else:
        forecast_trend = "Slightly Bearish ↘️"
        
    # Generate explicit trading signals based on indicators
    if rsi_14 < 30:
        rsi_signal = "OVERSOLD (Consider Buying)"
    elif rsi_14 > 70:
        rsi_signal = "OVERBOUGHT (Consider Selling)"
    else:
        rsi_signal = "NEUTRAL"
        
    if current_price > sma_20 and sma_20 > sma_50:
        tech_trend = "STRONG UPTREND"
        action = "BUY / HOLD 🟢"
    elif current_price < sma_20 and sma_20 < sma_50:
        tech_trend = "STRONG DOWNTREND"
        action = "SELL 🔴"
    else:
        tech_trend = "CONSOLIDATING"
        action = "WAIT 🟡"
        
    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "days_ahead": days_ahead,
        "forecast_trend": forecast_trend,
        "pct_change": pct_change,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "rsi_14": rsi_14,
        "rsi_signal": rsi_signal,
        "tech_trend": tech_trend,
        "action": action
    }
