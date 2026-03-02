import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_all_data(period: str = "5y") -> pd.DataFrame:
    """
    Fetches historical data for Gold futures, US Dollar Index, 10Y Treasury Yield, and S&P 500.
    Merges them into a single DataFrame.
    """
    logger.info(f"Fetching Gold, USD Index (DX-Y.NYB), 10Y Yield (^TNX), and S&P 500 (^GSPC) for {period}...")
    try:
        # Download all tickers
        tickers = ["GC=F", "DX-Y.NYB", "^TNX", "^GSPC"]
        
        # Download individually to handle potential yfinance grouped dataframe issues easily
        dfs = []
        for ticker in tickers:
            df_ticker = yf.download(ticker, period=period, progress=False)
            if df_ticker.empty:
                logger.warning(f"Fetched empty DataFrame for {ticker}")
                continue
            
            # yfinance sometimes returns MultiIndex columns if poorly formed, ensure we get a Series
            if isinstance(df_ticker.columns, pd.MultiIndex):
                close_series = df_ticker['Close'].iloc[:, 0]
            else:
                close_series = df_ticker['Close']
                
            name_map = {"GC=F": "Close", "DX-Y.NYB": "USD_Index", "^TNX": "Treasury_Yield", "^GSPC": "SP500"}
            close_series.name = name_map[ticker]
            dfs.append(close_series)
            
        if not dfs:
            return pd.DataFrame()
            
        # Merge them based on the Date index
        merged_df = pd.concat(dfs, axis=1)
        
        # Forward fill any missing days (e.g. if one market was closed but another was open)
        merged_df.ffill(inplace=True)
        # Drop rows where Gold was completely unavailable
        if 'Close' in merged_df.columns:
            merged_df.dropna(subset=['Close'], inplace=True)
            
        # Backward fill any remaining NaNs at the very beginning
        merged_df.bfill(inplace=True)
            
        logger.info(f"Successfully fetched and merged {len(merged_df)} records.")
        return merged_df
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

import pandas_ta as ta

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data specifically for Prophet.
    Prophet requires columns 'ds' (datestamp) and 'y' (target variable).
    Also calculates SMA_20, SMA_50, and RSI_14 as extra regressors.
    
    Args:
        df (pd.DataFrame): The raw DataFrame from yfinance.
        
    Returns:
        pd.DataFrame: A processed DataFrame with 'ds', 'y', 'SMA_20', 'SMA_50', and 'RSI_14' columns.
    """
    if df.empty:
        return df
        
    logger.info("Preprocessing data for Prophet and calculating indicators...")
    
    # Calculate indicators before dropping columns
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    
    # Ensure columns exist in case yfinance failed to fetch one of the macro regressors
    for col in ['USD_Index', 'Treasury_Yield', 'SP500']:
        if col not in df.columns:
            df[col] = 0.0
    
    processed_df = df.reset_index()[['Date', 'Close', 'SMA_20', 'SMA_50', 'RSI_14', 'USD_Index', 'Treasury_Yield', 'SP500']]
    
    # Ensure Date column is timezone un-aware to avoid Prophet warnings
    processed_df['Date'] = pd.to_datetime(processed_df['Date']).dt.tz_localize(None)
    
    processed_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    
    # Drop NaNs that exist at the start of the dataframe due to rolling windows (e.g., the first 50 days)
    processed_df.dropna(inplace=True)
    
    logger.info(f"Data preprocessed. Shape: {processed_df.shape}")
    return processed_df

if __name__ == "__main__":
    df = fetch_all_data(period="5y")
    print(df.head())
    
    processed = preprocess_data(df)
    print(processed.head())
