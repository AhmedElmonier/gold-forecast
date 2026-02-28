import argparse
import logging
import sys
from dotenv import load_dotenv

from src.data_fetcher import fetch_gold_data, preprocess_data
from src.model import GoldForecastModel, generate_insights
from src.alerter import format_alert_message, send_telegram_alert

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Gold Price Forecaster and Telegram Alerter")
    parser.add_argument("--dry-run", action="store_true", help="Run the pipeline without sending a Telegram message.")
    parser.add_argument("--days-ahead", type=int, default=30, help="Number of days to forecast into the future.")
    args = parser.parse_args()

    # Ensure environment variables are loaded for the alerter
    load_dotenv()
    
    logger.info("=== Starting Gold Forecasting Pipeline ===")
    
    # 1. Fetching Data
    logger.info("Step 1: Fetching data...")
    raw_df = fetch_gold_data(period="5y")
    if raw_df.empty:
        logger.error("Failed to fetch data. Exiting pipeline.")
        sys.exit(1)
        
    current_price = raw_df['Close'].iloc[-1]
    logger.info(f"Latest actual close price: ${current_price:.2f}")
    
    process_df = preprocess_data(raw_df)
    
    # 2. Forecasting
    logger.info("Step 2: Training Prophet model...")
    model = GoldForecastModel()
    model.fit(process_df)
    
    # Run cross-validation to get an idea of model performance
    model.evaluate()
    
    logger.info("Generating forecast...")
    forecast_df = model.predict(days_ahead=args.days_ahead)
    
    # 3. Generating Insights
    logger.info("Step 3: Generating insights...")
    insights = generate_insights(forecast_df, process_df, args.days_ahead)
    
    # 4. Alerting
    logger.info("Step 4: Formatting and sending alert...")
    msg = format_alert_message(insights)
    
    success = send_telegram_alert(msg, dry_run=args.dry_run)
    
    if success:
        logger.info("=== Pipeline finished successfully ===")
    else:
        logger.error("=== Pipeline finished with errors ===")

if __name__ == "__main__":
    main()
