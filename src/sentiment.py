import yfinance as yf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger(__name__)

# Ensure the VADER lexicon is downloaded (only downloads if not present)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("Downloading NLTK VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)

def analyze_gold_headlines() -> dict:
    """
    Fetches the latest news headlines for Gold from Yahoo Finance
    and calculates an average sentiment score using NLTK VADER.
    
    Returns:
        dict: Containing 'score' (-1 to 1), 'label' (String), and 'article_count'.
    """
    logger.info("Fetching latest news headlines for Gold (GC=F)...")
    try:
        gold = yf.Ticker("GC=F")
        news = gold.news
        
        if not news:
            logger.warning("No recent news found for Gold.")
            return {"score": 0.0, "label": "No Data", "article_count": 0}
            
        sia = SentimentIntensityAnalyzer()
        total_score = 0
        valid_articles = 0
        
        for article in news:
            title = article.get('title', '')
            if title:
                # Calculate compound score (-1 to +1) for the headline
                sentiment = sia.polarity_scores(title)
                total_score += sentiment['compound']
                valid_articles += 1
                
        if valid_articles == 0:
            return {"score": 0.0, "label": "No Data", "article_count": 0}
            
        avg_score = total_score / valid_articles
        
        # Determine human-readable label
        if avg_score > 0.2:
            label = "Optimistic 🟢"
        elif avg_score < -0.2:
            label = "Pessimistic 🔴"
        else:
            label = "Neutral 🟡"
            
        logger.info(f"Analyzed {valid_articles} headlines. Average Sentiment: {avg_score:.2f} ({label})")
        
        return {
            "score": avg_score,
            "label": label,
            "article_count": valid_articles
        }
        
    except Exception as e:
        logger.error(f"Error fetching/analyzing news sentiment: {e}")
        return {"score": 0.0, "label": "Error", "article_count": 0}

def get_detailed_news_sentiment(limit: int = 5) -> dict:
    """
    Fetches the latest news headlines for Gold from Yahoo Finance
    and calculates sentiment score using NLTK VADER for each article.
    
    Returns:
        dict: Containing 'overall_score', 'overall_label', 'article_count', and a list of 'articles'
              each with 'title', 'link', 'score', and 'label'.
    """
    logger.info(f"Fetching latest {limit} news headlines for Gold (GC=F)...")
    try:
        gold = yf.Ticker("GC=F")
        news = gold.news
        
        if not news:
            logger.warning("No recent news found for Gold.")
            return {"overall_score": 0.0, "overall_label": "No Data", "article_count": 0, "articles": []}
            
        sia = SentimentIntensityAnalyzer()
        total_score = 0
        valid_articles = 0
        detailed_articles = []
        
        for article in news[:limit]:
            title = article.get('title', '')
            link = article.get('link', '')
            if title:
                sentiment = sia.polarity_scores(title)
                score = sentiment['compound']
                
                total_score += score
                valid_articles += 1
                
                if score > 0.2:
                    label = "Optimistic 🟢"
                elif score < -0.2:
                    label = "Pessimistic 🔴"
                else:
                    label = "Neutral 🟡"
                    
                detailed_articles.append({
                    "title": title,
                    "link": link,
                    "score": score,
                    "label": label
                })
                
        if valid_articles == 0:
            return {"overall_score": 0.0, "overall_label": "No Data", "article_count": 0, "articles": []}
            
        avg_score = total_score / valid_articles
        
        # Determine human-readable label
        if avg_score > 0.2:
            overall_label = "Optimistic 🟢"
        elif avg_score < -0.2:
            overall_label = "Pessimistic 🔴"
        else:
            overall_label = "Neutral 🟡"
            
        return {
            "overall_score": avg_score,
            "overall_label": overall_label,
            "article_count": valid_articles,
            "articles": detailed_articles
        }
        
    except Exception as e:
        logger.error(f"Error fetching/analyzing detailed news sentiment: {e}")
        return {"overall_score": 0.0, "overall_label": "Error", "article_count": 0, "articles": []}

if __name__ == "__main__":
    result = analyze_gold_headlines()
    print(f"Sentiment Result: {result}")
