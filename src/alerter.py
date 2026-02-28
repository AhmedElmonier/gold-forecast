import os
import requests
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def format_alert_message(insights: dict) -> str:
    """
    Formats the insights dictionary into a Markdown-friendly Telegram message.
    """
    msg = f"🏆 *Gold Price Forecast & Insights ({insights['days_ahead']} days)* 🏆\n\n"
    
    msg += f"🔥 *Trading Signals (Live)* 🔥\n"
    msg += f"• *Action:* {insights['action']}\n"
    msg += f"• *Trend:* {insights['tech_trend']}\n"
    msg += f"• *Momentum (RSI):* {insights['rsi_signal']}\n\n"
    
    msg += f"💵 *Current Price:* ${insights['current_price']:.2f}\n"
    msg += f"📉 *20-Day SMA:* ${insights['sma_20']:.2f} | *50-Day SMA:* ${insights['sma_50']:.2f}\n"
    msg += f"🌡️ *14-Day RSI:* {insights['rsi_14']:.2f}\n\n"
    
    msg += f"🔮 *Future Prophet Forecast*\n"
    msg += f"• *Predicted Price:* ${insights['predicted_price']:.2f}\n"
    msg += f"• *Lower Bound:* ${insights['lower_bound']:.2f} | *Upper Bound:* ${insights['upper_bound']:.2f}\n"
    msg += f"• *Expected Change:* {insights['pct_change']:.2f}%\n"
    msg += f"• *Forecast Trend:* {insights['forecast_trend']}\n"
    
    return msg

def send_telegram_alert(message: str, dry_run: bool = False) -> bool:
    """
    Sends a formatted Markdown message to the configured Telegram chat.
    
    Args:
        message (str): The Markdown formatted message.
        dry_run (bool): If True, only logs the message without sending it.
        
    Returns:
        bool: True if sent successfully (or if dry_run), False otherwise.
    """
    if dry_run:
        logger.info("\n--- DRY RUN: Telegram Alert Message ---\n" + message + "\n---------------------------------------")
        return True
        
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing! Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Successfully sent Telegram alert.")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error sending Telegram alert: {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error sending Telegram alert: {e}")
        return False
