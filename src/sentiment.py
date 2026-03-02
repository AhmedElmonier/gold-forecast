import yfinance as yf
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Initialize the FinBERT sentiment pipeline
try:
    logger.info("Initializing FinBERT sentiment pipeline... (This may take a moment to load the model)")
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    logger.error(f"Failed to initialize FinBERT: {e}")
    sentiment_pipeline = None

def analyze_gold_headlines() -> dict:
    """
    Fetches the latest news headlines for Gold from Yahoo Finance
    and calculates an average sentiment score using FinBERT.
    
    Returns:
        dict: Containing 'score' (mapped roughly to -1 to 1), 'label' (String), and 'article_count'.
    """
    logger.info("Fetching latest news headlines for Gold (GC=F)...")
    if not sentiment_pipeline:
        return {"score": 0.0, "label": "Model Error", "article_count": 0}
        
    try:
        gold = yf.Ticker("GC=F")
        news = gold.news
        
        if not news:
            logger.warning("No recent news found for Gold.")
            return {"score": 0.0, "label": "No Data", "article_count": 0}
            
        total_score = 0
        valid_articles = 0
        
        for article in news:
            title = article.get('title', '')
            if title:
                # FinBERT returns e.g. [{'label': 'positive', 'score': 0.85}]
                result = sentiment_pipeline(title)[0]
                label = result['label']
                conf = result['score']
                
                # Map FinBERT labels to a numeric score (-1 to 1)
                num_score = conf if label == 'positive' else (-conf if label == 'negative' else 0.0)
                total_score += num_score
                valid_articles += 1
                
        if valid_articles == 0:
            return {"score": 0.0, "label": "No Data", "article_count": 0}
            
        avg_score = total_score / valid_articles
        
        # Determine aggregate human-readable label
        if avg_score > 0.15:
            label = "Optimistic 🟢"
        elif avg_score < -0.15:
            label = "Pessimistic 🔴"
        else:
            label = "Neutral 🟡"
            
        logger.info(f"Analyzed {valid_articles} headlines with FinBERT. Average Sentiment: {avg_score:.2f} ({label})")
        
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
    and calculates sentiment score using FinBERT for each article.
    
    Returns:
        dict: Containing 'overall_score', 'overall_label', 'article_count', and a list of 'articles'
              each with 'title', 'link', 'score', and 'label'.
    """
    logger.info(f"Fetching latest {limit} news headlines for Gold (GC=F)...")
    if not sentiment_pipeline:
        return {"overall_score": 0.0, "overall_label": "Model Error", "article_count": 0, "articles": []}

    try:
        gold = yf.Ticker("GC=F")
        news = gold.news
        
        if not news:
            logger.warning("No recent news found for Gold.")
            return {"overall_score": 0.0, "overall_label": "No Data", "article_count": 0, "articles": []}
            
        total_score = 0
        valid_articles = 0
        detailed_articles = []
        
        for article in news[:limit]:
            title = article.get('title', '')
            link = article.get('link', '')
            if title:
                result = sentiment_pipeline(title)[0]
                label_txt = result['label']
                conf = result['score']
                
                num_score = conf if label_txt == 'positive' else (-conf if label_txt == 'negative' else 0.0)
                
                total_score += num_score
                valid_articles += 1
                
                if label_txt == 'positive':
                    label = "Optimistic 🟢"
                elif label_txt == 'negative':
                    label = "Pessimistic 🔴"
                else:
                    label = "Neutral 🟡"
                    
                detailed_articles.append({
                    "title": title,
                    "link": link,
                    "score": num_score,
                    "label": label
                })
                
        if valid_articles == 0:
            return {"overall_score": 0.0, "overall_label": "No Data", "article_count": 0, "articles": []}
            
        avg_score = total_score / valid_articles
        
        # Determine aggregate human-readable label
        if avg_score > 0.15:
            overall_label = "Optimistic 🟢"
        elif avg_score < -0.15:
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
