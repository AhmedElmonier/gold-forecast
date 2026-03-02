import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

from src.data_fetcher import fetch_all_data, preprocess_data
from src.model import GoldForecastModel, XGBoostForecaster, generate_insights
from src.alerter import format_alert_message
from src.charting import generate_forecast_chart
from src.sentiment import analyze_headlines, get_detailed_news_sentiment
from src.portfolio import execute_trade, get_balance, get_portfolio
from src.correlation import calculate_correlations, get_correlation_insights
from src.llm import generate_daily_brief

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
        "📰 `/news` - Get the latest Gold headlines with individual FinBERT sentiment analysis.\n"
        "📊 `/stats` - View historical backtest performance of the trading strategy.\n"
        "📝 `/brief` - Read an AI-generated daily market summary.\n"
        "💼 `/buy`, `/sell`, `/portfolio` - Paper trading commands.\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch the latest price and indicators instantly without running the Prophet model."""
    ticker = "GC=F"
    if context.args:
        ticker = context.args[0].upper()
        
    await update.message.reply_text(f"⏳ Fetching real-time market data for {ticker}... Please wait.")
    try:
        raw_df = fetch_all_data(primary_ticker=ticker, period="5y")
        if raw_df.empty:
            await update.message.reply_text(f"❌ Failed to fetch data for {ticker}. Yahoo Finance might be down or the ticker is invalid.")
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
            
        sentiment = analyze_gold_headlines() if ticker == "GC=F" else {"label": "N/A", "score": 0.0}
            
        msg = f"🔥 *Real-Time Trading Signals for {ticker}* 🔥\n\n"
        msg += f"💵 *Current Price:* ${current_price:.2f}\n\n"
        msg += f"• *Action:* {action}\n"
        msg += f"• *Trend:* {tech_trend}\n"
        msg += f"• *Momentum (RSI):* {rsi_signal} (Value: {rsi_14:.2f})\n"
        
        if ticker == "GC=F":
            msg += f"• *News Sentiment:* {sentiment['label']} (Score: {sentiment['score']:.2f})\n\n"
        else:
            msg += "\n"
            
        msg += f"📉 *20-Day SMA:* ${sma_20:.2f} | *50-Day SMA:* ${sma_50:.2f}\n"

        await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /price command: {e}")
        await update.message.reply_text(f"❌ An error occurred: {str(e)}")

async def forecast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the full Prophet pipeline, generate a chart, and send it back."""
    ticker = "GC=F"
    if context.args:
        ticker = context.args[0].upper()
        
    await update.message.reply_text(f"🤖 Initializing Prophet AI Model for {ticker}. Fetching 5 years of historical data and building future projections. This will take roughly 15-20 seconds...")
    
    try:
        raw_df = fetch_all_data(primary_ticker=ticker, period="5y")
        if raw_df.empty:
            await update.message.reply_text(f"❌ Failed to fetch data for {ticker}.")
            return
            
        process_df = preprocess_data(raw_df)
        
        model = GoldForecastModel()
        model.fit(process_df)
        forecast_df = model.predict(process_df, days_ahead=30)
        
        xgb_model = XGBoostForecaster()
        xgb_model.fit(process_df, days_ahead=30)
        xgb_pred = xgb_model.predict_current(process_df)
        
        insights = generate_insights(forecast_df, process_df, days_ahead=30, xgb_prediction=xgb_pred)
        
        sentiment = analyze_headlines(ticker)
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

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch the latest news and detailed sentiment. Usage: /news [ticker]"""
    ticker = "GC=F"
    if context.args:
        ticker = context.args[0].upper()
        
    await update.message.reply_text(f"📰 Fetching latest news and analyzing sentiment for {ticker}...")
    try:
        sentiment_data = get_detailed_news_sentiment(ticker=ticker, limit=5)
        
        if sentiment_data['article_count'] == 0:
            await update.message.reply_text(f"❌ No recent news found for {ticker}.")
            return
            
        msg = f"📰 *Latest News Sentiment for {ticker}* 📰\n\n"
        msg += f"📊 *Overall Sentiment:* {sentiment_data['overall_label']} (Score: {sentiment_data['overall_score']:.2f})\n\n"
        
        for idx, article in enumerate(sentiment_data['articles'], 1):
            msg += f"*{idx}. {article['title']}*\n"
            msg += f"• *Sentiment:* {article['label']} (Score: {article['score']:.2f})\n"
            msg += f"• [Read Article]({article['link']})\n\n"
            
        await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=True)

    except Exception as e:
        logger.error(f"Error in /news command: {e}")
        await update.message.reply_text(f"❌ An error occurred: {str(e)}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the backtest and return historical performance metrics."""
    await update.message.reply_text("📊 Running historical backtest simulation. This might take a few seconds...")
    try:
        from src.backtest import run_backtest
        stats_msg = run_backtest()
        await update.message.reply_text(stats_msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in /stats command: {e}")
        await update.message.reply_text(f"❌ An error occurred while running backtest: {str(e)}")

async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Execute a virtual buy trade. Usage: /buy <quantity> [ticker]"""
    if not context.args:
        await update.message.reply_text("❌ Please specify a quantity. Usage: `/buy <quantity> [ticker]`", parse_mode="Markdown")
        return
        
    try:
        quantity = float(context.args[0])
        ticker = context.args[1].upper() if len(context.args) > 1 else "GC=F"
        user_id = update.effective_user.id
        
        # Fetch current price
        raw_df = fetch_all_data(primary_ticker=ticker, period="1d")
        if raw_df.empty:
            await update.message.reply_text(f"❌ Failed to fetch current price for {ticker}.")
            return
            
        current_price = float(raw_df['Close'].iloc[-1])
        
        success, msg = execute_trade(user_id, ticker, quantity, "BUY", current_price)
        await update.message.reply_text(msg)
        
    except ValueError:
        await update.message.reply_text("❌ Invalid quantity. Please enter a number.")
    except Exception as e:
        logger.error(f"Error in /buy command: {e}")
        await update.message.reply_text(f"❌ An error occurred: {str(e)}")

async def sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Execute a virtual sell trade. Usage: /sell <quantity> [ticker]"""
    if not context.args:
        await update.message.reply_text("❌ Please specify a quantity. Usage: `/sell <quantity> [ticker]`", parse_mode="Markdown")
        return
        
    try:
        quantity = float(context.args[0])
        ticker = context.args[1].upper() if len(context.args) > 1 else "GC=F"
        user_id = update.effective_user.id
        
        # Fetch current price
        raw_df = fetch_all_data(primary_ticker=ticker, period="1d")
        if raw_df.empty:
            await update.message.reply_text(f"❌ Failed to fetch current price for {ticker}.")
            return
            
        current_price = float(raw_df['Close'].iloc[-1])
        
        success, msg = execute_trade(user_id, ticker, quantity, "SELL", current_price)
        await update.message.reply_text(msg)
        
    except ValueError:
        await update.message.reply_text("❌ Invalid quantity. Please enter a number.")
    except Exception as e:
        logger.error(f"Error in /sell command: {e}")
        await update.message.reply_text(f"❌ An error occurred: {str(e)}")

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the user's virtual portfolio."""
    user_id = update.effective_user.id
    balance = get_balance(user_id)
    holdings = get_portfolio(user_id)
    
    msg = f"💼 *Virtual Portfolio* 💼\n\n"
    msg += f"💵 *Available Cash:* ${balance:,.2f}\n\n"
    
    if not holdings:
        msg += "You don't own any assets yet. Use `/buy <qty> [ticker]` to start trading!"
        await update.message.reply_text(msg, parse_mode="Markdown")
        return
        
    msg += "*Current Holdings:*\n"
    total_value = balance
    
    # Send an initial message since fetching prices might take a second
    status_message = await update.message.reply_text("⏳ Calculating portfolio value...")
    
    for ticker, data in holdings.items():
        qty = data['quantity']
        avg_price = data['avg_price']
        
        # Fetch current price
        raw_df = fetch_all_data(primary_ticker=ticker, period="1d")
        if not raw_df.empty:
            current_price = float(raw_df['Close'].iloc[-1])
            value = qty * current_price
            total_value += value
            
            pl_per_unit = current_price - avg_price
            pl_total = pl_per_unit * qty
            pl_pct = (pl_per_unit / avg_price) * 100
            
            emoji = "🟢" if pl_total >= 0 else "🔴"
            msg += f"• *{ticker}*: {qty} units\n"
            msg += f"  Value: ${value:,.2f}\n"
            msg += f"  P/L: {emoji} ${pl_total:,.2f} ({pl_pct:.2f}%)\n\n"
        else:
            msg += f"• *{ticker}*: {qty} units (Could not fetch current price)\n\n"
            
    msg += f"💰 *Total Account Value:* ${total_value:,.2f}"
    
    await status_message.edit_text(msg, parse_mode="Markdown")

async def brief_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a daily market brief using an LLM."""
    ticker = "GC=F"
    if context.args:
        ticker = context.args[0].upper()
        
    await update.message.reply_text(f"🤖 Analyzing market conditions for {ticker} and generating a brief... This may take up to 30 seconds.")
    
    try:
        raw_df = fetch_all_data(primary_ticker=ticker, period="5y")
        if raw_df.empty:
            await update.message.reply_text(f"❌ Failed to fetch data for {ticker}.")
            return
            
        process_df = preprocess_data(raw_df)
        
        # Get forecast and technicals
        model = GoldForecastModel()
        model.fit(process_df)
        forecast_df = model.predict(process_df, days_ahead=30)
        
        xgb_model = XGBoostForecaster()
        xgb_model.fit(process_df, days_ahead=30)
        xgb_pred = xgb_model.predict_current(process_df)
        
        insights = generate_insights(forecast_df, process_df, days_ahead=30, xgb_prediction=xgb_pred)
        current_price = insights['current_price']
        
        # Get Sentiment
        sentiment = analyze_headlines(ticker)
        
        # Get Correlations
        correlations = calculate_correlations(process_df, primary_col='y')
        
        # Generate LLM Brief
        brief_text = generate_daily_brief(
            ticker=ticker,
            current_price=current_price,
            forecast_data=insights,
            sentiment_data=sentiment,
            correlation_data=correlations
        )
        
        await update.message.reply_text(brief_text, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error in /brief command: {e}")
        await update.message.reply_text(f"❌ An error occurred during brief generation: {str(e)}")

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
    application.add_handler(CommandHandler("news", news_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("buy", buy_command))
    application.add_handler(CommandHandler("sell", sell_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    application.add_handler(CommandHandler("brief", brief_command))

    # Run the bot until the user presses Ctrl-C
    logger.info("Bot is polling... Listening for commands.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
