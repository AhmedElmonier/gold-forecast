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

import feedparser

def fetch_broad_market_news(limit: int = 15) -> list:
    """
    Fetches news from broader economic/geopolitical RSS feeds.
    """
    rss_urls = [
        "https://finance.yahoo.com/news/rssindex",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664" # CNBC Finance
    ]
    
    articles = []
    
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                articles.append({
                    "title": entry.title,
                    "link": entry.link
                })
        except Exception as e:
            logger.warning(f"Failed to fetch RSS feed {url}: {e}")
            
    # Remove duplicates
    unique_articles = {article["title"]: article for article in articles}.values()
    return list(unique_articles)[:limit]

def analyze_headlines(ticker: str = "GC=F") -> dict:
    """
    Fetches the latest news headlines and calculates an average sentiment score using FinBERT.
    For Gold (GC=F), intertwines broad economic news.
    """
    logger.info(f"Fetching latest news headlines for {ticker}...")
    if not sentiment_pipeline:
        return {"score": 0.0, "label": "Model Error", "article_count": 0}
        
    try:
        asset = yf.Ticker(ticker)
        news = asset.news or []
        
        articles = [{"title": a.get("title", ""), "link": a.get("link", "")} for a in news if a.get("title")]
        
        if ticker in ["GC=F", "SI=F"]:
            logger.info("Adding broader economic context for precious metals...")
            articles.extend(fetch_broad_market_news(limit=10))
            
        if not articles:
            logger.warning(f"No recent news found for {ticker}.")
            return {"score": 0.0, "label": "No Data", "article_count": 0}
            
        total_score = 0
        valid_articles = 0
        
        for article in articles:
            title = article['title']
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

def get_detailed_news_sentiment(ticker: str = "GC=F", limit: int = 5) -> dict:
    """
    Fetches the latest news headlines and calculates sentiment score using FinBERT for each article.
    """
    logger.info(f"Fetching latest {limit} news headlines for {ticker}...")
    if not sentiment_pipeline:
        return {"overall_score": 0.0, "overall_label": "Model Error", "article_count": 0, "articles": []}

    try:
        asset = yf.Ticker(ticker)
        news = asset.news or []
        
        articles = [{"title": a.get("title", ""), "link": a.get("link", "")} for a in news if a.get("title")]
        
        if ticker in ["GC=F", "SI=F"]:
            articles.extend(fetch_broad_market_news(limit=10))
            
        if not articles:
            logger.warning(f"No recent news found for {ticker}.")
            return {"overall_score": 0.0, "overall_label": "No Data", "article_count": 0, "articles": []}
            
        total_score = 0
        valid_articles = 0
        detailed_articles = []
        
        # We only want to detail 'limit' articles to not spam the user
        for article in articles[:limit]:
            title = article['title']
            link = article['link']
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
