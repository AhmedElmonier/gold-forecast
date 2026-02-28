import logging
from src.bot import main as bot_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Gold Forecast Interactive Telegram Bot...")
    bot_main()
