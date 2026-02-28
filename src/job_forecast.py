import os
import sys
import logging

# Add the parent directory (project root) to sys.path so we can import 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.data_fetcher import fetch_all_data, preprocess_data
from src.model import GoldForecastModel, generate_insights
from src.alerter import format_alert_message, send_telegram_alert
from src.charting import generate_forecast_chart
from src.sentiment import analyze_gold_headlines

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_scheduled_job():
    """Runs a single pass of the forecasting pipeline and sends an alert."""
    load_dotenv()
    
    logger.info("=== Starting Automated Gold Forecasting Job ===")
    
    logger.info("Step 1: Fetching data...")
    raw_df = fetch_all_data(period="5y")
    if raw_df.empty:
        logger.error("Failed to fetch data. Exiting.")
        sys.exit(1)
        
    process_df = preprocess_data(raw_df)
    
    logger.info("Step 2: Training Prophet model...")
    model = GoldForecastModel()
    model.fit(process_df)
    
    logger.info("Generating forecast...")
    forecast_df = model.predict(process_df, days_ahead=30)
    
    logger.info("Step 3: Generating insights...")
    insights = generate_insights(forecast_df, process_df, 30)
    
    logger.info("Step 3.5: Analyzing news sentiment...")
    sentiment = analyze_gold_headlines()
    insights['sentiment_label'] = sentiment['label']
    insights['sentiment_score'] = sentiment['score']
    insights['sentiment_count'] = sentiment['article_count']
    
    logger.info("Step 4: Generating visual chart...")
    chart_path = generate_forecast_chart(process_df, forecast_df)
    
    logger.info("Step 5: Formatting and sending alert...")
    msg = format_alert_message(insights)
    
    success = send_telegram_alert(msg, image_path=chart_path)
    
    if success:
        logger.info("=== Job finished successfully ===")
    else:
        logger.error("=== Job finished with errors ===")

if __name__ == "__main__":
    run_scheduled_job()
