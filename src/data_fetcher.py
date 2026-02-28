import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_all_data(period: str = "5y") -> pd.DataFrame:
    """
    Fetches historical data for Gold futures, US Dollar Index, and 10Y Treasury Yield.
    Merges them into a single DataFrame.
    """
    logger.info(f"Fetching Gold (GC=F), USD Index (DX-Y.NYB), and 10Y Yield (^TNX) for the last {period}...")
    try:
        # Download all three tickers at once
        tickers = "GC=F DX-Y.NYB ^TNX"
        df = yf.download(tickers, period=period, group_by='ticker', progress=False)
        
        if df.empty:
            logger.warning("Fetched an empty DataFrame.")
            return pd.DataFrame()
            
        # Extract the 'Close' prices for each asset
        gold_close = df['GC=F']['Close'].rename('Close')
        dxy_close = df['DX-Y.NYB']['Close'].rename('USD_Index')
        tnx_close = df['^TNX']['Close'].rename('Treasury_Yield')
        
        # Merge them based on the Date index
        merged_df = pd.concat([gold_close, dxy_close, tnx_close], axis=1)
        
        # Forward fill any missing days (e.g. if one market was closed but another was open)
        merged_df.ffill(inplace=True)
        # Drop rows where Gold was completely unavailable
        merged_df.dropna(subset=['Close'], inplace=True)
        
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
    
    processed_df = df.reset_index()[['Date', 'Close', 'SMA_20', 'SMA_50', 'RSI_14', 'USD_Index', 'Treasury_Yield']]
    
    # Ensure Date column is timezone un-aware to avoid Prophet warnings
    processed_df['Date'] = pd.to_datetime(processed_df['Date']).dt.tz_localize(None)
    
    processed_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    
    # Drop NaNs that exist at the start of the dataframe due to rolling windows (e.g., the first 50 days)
    processed_df.dropna(inplace=True)
    
    logger.info(f"Data preprocessed. Shape: {processed_df.shape}")
    return processed_df

if __name__ == "__main__":
    df = fetch_gold_data(period="5y")
    print(df.head())
    
    processed = preprocess_data(df)
    print(processed.head())
