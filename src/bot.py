import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

from src.data_fetcher import fetch_all_data, preprocess_data
from src.model import GoldForecastModel, generate_insights
from src.alerter import format_alert_message
from src.charting import generate_forecast_chart
from src.sentiment import analyze_gold_headlines

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    welcome_text = (
        f"Hi {user.first_name}! 👋\n\n"
        "I am your automated Gold AI Assistant. I monitor the markets 24/7 so you don't have to.\n\n"
        "Here is what I can do:\n"
        "📉 `/price` - Get the live price, real-time technical indicators, and Buy/Sell signals.\n"
        "🔮 `/forecast` - Run a full AI simulation to predict the price 30 days into the future (takes ~15 seconds).\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch the latest price and indicators instantly without running the Prophet model."""
    await update.message.reply_text("⏳ Fetching real-time market data... Please wait.")
    try:
        raw_df = fetch_all_data(period="5y")
        if raw_df.empty:
            await update.message.reply_text("❌ Failed to fetch data. Yahoo Finance might be down.")
            return
            
        process_df = preprocess_data(raw_df)
        
        # We can borrow the logic from generate_insights by mocking a forecast
        # just to format the technical indicator signal block (Action/Trend/RSI)
        latest_hist = process_df.iloc[-1]
        current_price = latest_hist['y']
        sma_20 = latest_hist['SMA_20']
        sma_50 = latest_hist['SMA_50']
        rsi_14 = latest_hist['RSI_14']
        
        if rsi_14 < 30:
            rsi_signal = "OVERSOLD (Consider Buying) 🟢"
        elif rsi_14 > 70:
            rsi_signal = "OVERBOUGHT (Consider Selling) 🔴"
        else:
            rsi_signal = "NEUTRAL 🟡"
            
        if current_price > sma_20 and sma_20 > sma_50:
            tech_trend = "STRONG UPTREND 📈"
            action = "BUY / HOLD 🟢"
        elif current_price < sma_20 and sma_20 < sma_50:
            tech_trend = "STRONG DOWNTREND 📉"
            action = "SELL 🔴"
        else:
            tech_trend = "CONSOLIDATING ➖"
            action = "WAIT 🟡"
            
        sentiment = analyze_gold_headlines()
            
        msg = f"🔥 *Real-Time Trading Signals* 🔥\n\n"
        msg += f"💵 *Current Price:* ${current_price:.2f}\n\n"
        msg += f"• *Action:* {action}\n"
        msg += f"• *Trend:* {tech_trend}\n"
        msg += f"• *Momentum (RSI):* {rsi_signal} (Value: {rsi_14:.2f})\n"
        msg += f"• *News Sentiment:* {sentiment['label']} (Score: {sentiment['score']:.2f})\n\n"
        msg += f"📉 *20-Day SMA:* ${sma_20:.2f} | *50-Day SMA:* ${sma_50:.2f}\n"

        await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /price command: {e}")
        await update.message.reply_text(f"❌ An error occurred: {str(e)}")

async def forecast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the full Prophet pipeline, generate a chart, and send it back."""
    await update.message.reply_text("🤖 Initializing Prophet AI Model. Fetching 5 years of historical data and building future projections. This will take roughly 15-20 seconds...")
    
    try:
        raw_df = fetch_all_data(period="5y")
        if raw_df.empty:
            await update.message.reply_text("❌ Failed to fetch data.")
            return
            
        process_df = preprocess_data(raw_df)
        
        model = GoldForecastModel()
        model.fit(process_df)
        
        forecast_df = model.predict(process_df, days_ahead=30)
        insights = generate_insights(forecast_df, process_df, days_ahead=30)
        
        sentiment = analyze_gold_headlines()
        insights['sentiment_label'] = sentiment['label']
        insights['sentiment_score'] = sentiment['score']
        insights['sentiment_count'] = sentiment['article_count']
        
        msg = format_alert_message(insights)
        chart_path = generate_forecast_chart(process_df, forecast_df)
        
        if os.path.exists(chart_path):
            with open(chart_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=msg, parse_mode="Markdown")
        else:
            # Fallback if chart generation failed
            await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /forecast command: {e}")
        await update.message.reply_text(f"❌ An error occurred during forecasting: {str(e)}")

def main() -> None:
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in .env file!")
        return

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("forecast", forecast_command))

    # Run the bot until the user presses Ctrl-C
    logger.info("Bot is polling... Listening for commands.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
