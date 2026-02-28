import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_gold_data(period: str = "10y") -> pd.DataFrame:
    """
    Fetches historical gold futures data from Yahoo Finance.
    
    Args:
        period (str): The period of data to fetch (e.g., '10y', '5y', '1y').
        
    Returns:
        pd.DataFrame: A DataFrame containing the historical data.
    """
    logger.info(f"Fetching Gold data (GC=F) for the last {period}...")
    try:
        gold = yf.Ticker("GC=F")
        df = gold.history(period=period)
        
        if df.empty:
            logger.warning("Fetched an empty DataFrame. Please check the ticker symbol or network connection.")
            return df
            
        logger.info(f"Successfully fetched {len(df)} records.")
        return df
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
    
    processed_df = df.reset_index()[['Date', 'Close', 'SMA_20', 'SMA_50', 'RSI_14']]
    
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
