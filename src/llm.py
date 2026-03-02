import os
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def generate_daily_brief(ticker: str, current_price: float, forecast_data: dict, sentiment_data: dict, correlation_data: dict) -> str:
    """
    Constructs a nuanced, human-like market brief using the Google GenAI API.
    """
    logger.info("Generating LLM Daily Brief...")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("No LLM_API_KEY or GEMINI_API_KEY found in environment.")
        return "❌ Error: LLM API Key not configured. Please add LLM_API_KEY to your .env file."
        
    try:
        # Initialize the new Google GenAI SDK client
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are an expert financial analyst writing a concise, engaging daily brief for a Telegram trading channel.
        Write a summary analyzing the current state of {ticker}. 
        Keep it under 200 words. Use formatting like bolding and bullet points where appropriate, and a few relevant emojis.
        
        Here is the latest data to base your analysis on:
        - Current Price: ${current_price:.2f}
        - 30-Day Forecast Trend: {forecast_data.get('forecast_trend', 'Unknown')}
        - Forecasted 30-Day Target Price: ${forecast_data.get('predicted_price', 0):.2f}
        - Technical Action Signal: {forecast_data.get('action', 'Unknown')}
        - Technical Trend: {forecast_data.get('tech_trend', 'Unknown')}
        - Latest News Sentiment: {sentiment_data.get('overall_label', 'Unknown')} (Score: {sentiment_data.get('overall_score', 0):.2f} based on {sentiment_data.get('article_count', 0)} articles)
        
        Recent Correlations:
        {correlation_data}
        
        Synthesize this data. Don't just list the numbers back; explain what the combination of technicals, sentiment, and correlations might mean for a trader right now.
        End with a quick disclaimer that trading involves risk.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating LLM brief: {e}")
        return f"❌ Failed to generate LLM brief. Error: {str(e)}"
