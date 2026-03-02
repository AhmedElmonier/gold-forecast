import pandas as pd
import logging
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

def calculate_correlations(df: pd.DataFrame, primary_col: str = 'y') -> dict:
    """
    Calculates Pearson correlation between the primary asset and regressors 
    over the last 30 and 90 days.
    
    Args:
        df: DataFrame containing the 'y' (primary asset) and regressor columns.
        primary_col: The name of the column representing the primary asset's price.
        
    Returns:
        A dictionary containing the correlation values.
    """
    logger.info("Computing correlation matrices...")
    
    correlations = {
        "30d": {},
        "90d": {}
    }
    
    # We only care about correlating the primary asset against our macroeconomic features
    features = ['USD_Index', 'Treasury_Yield', 'SP500']
    
    # Check if we have enough data
    if len(df) < 90:
        logger.warning("Not enough data to calculate 90-day correlations.")
        return correlations
        
    df_30 = df.tail(30).dropna(subset=[primary_col] + features)
    df_90 = df.tail(90).dropna(subset=[primary_col] + features)
    
    for feature in features:
        if feature in df_30.columns:
            r_30, p_30 = pearsonr(df_30[primary_col], df_30[feature])
            correlations["30d"][feature] = r_30
            
        if feature in df_90.columns:
            r_90, p_90 = pearsonr(df_90[primary_col], df_90[feature])
            correlations["90d"][feature] = r_90
            
    logger.info(f"Computed correlations: {correlations}")
    return correlations

def get_correlation_insights(correlations: dict) -> list:
    """
    Translates correlation coefficients into human-readable insights.
    """
    insights = []
    
    for period in ["30d", "90d"]:
        if period not in correlations or not correlations[period]:
            continue
            
        period_str = "1-Month" if period == "30d" else "3-Month"
        
        for feature, r_val in correlations[period].items():
            if r_val > 0.7:
                strength = "Strong Positive Correlation"
            elif r_val > 0.3:
                strength = "Moderate Positive Correlation"
            elif r_val < -0.7:
                strength = "Strong Negative Correlation"
            elif r_val < -0.3:
                strength = "Moderate Negative Correlation"
            else:
                strength = "Weak/No Correlation"
                
            insights.append(f"• *{feature}* ({period_str}): {strength} (r={r_val:.2f})")
            
    return insights
