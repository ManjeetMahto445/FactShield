"""
FactShield — Latest News Service
Author: FactShield Development Team
Description: Fetches and processes recent news articles using DuckDuckGo search.
Provides categories, custom searches, and formats output into clean Pandas DataFrames.
"""

import pandas as pd
from utils import clean_text, format_date, get_domain_name

# Resilient DDGS import (supports both 'ddgs' and legacy 'duckduckgo_search')
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Predefined categories for one-click exploration
NEWS_CATEGORIES = {
    "🌐 Top Stories": "breaking news world",
    "💻 Technology": "technology innovation AI computing",
    "🔬 Science": "science research space discovery",
    "💼 Business": "economy market business finance",
    "🏥 Health": "health medicine wellness medical",
    "🌱 Environment": "climate environment renewable energy",
}


def fetch_latest_news(query: str = "breaking news", max_results: int = 10, timelimit: str = "w") -> pd.DataFrame:
    """
    Fetches recent news articles based on query or category.
    
    Args:
        query: Search keywords or topic.
        max_results: Maximum number of articles to return (5 to 30).
        timelimit: Time window ('d' for past day, 'w' for past week, 'm' for past month).
        
    Returns:
        Pandas DataFrame containing cleaned news items with standardized columns:
        ['title', 'source', 'date', 'formatted_date', 'snippet', 'url', 'image']
    """
    if not query or not query.strip():
        query = "world news"

    articles = []
    
    try:
        ddgs = DDGS()
        
        # Primary attempt: DuckDuckGo dedicated news API
        raw_news = list(ddgs.news(query.strip(), max_results=max_results, timelimit=timelimit))
        
        # Fallback to standard web text search if news endpoint returned 0 items
        if not raw_news:
            raw_text = list(ddgs.text(f"{query.strip()} news", max_results=max_results))
            for item in raw_text:
                url = item.get("href") or item.get("url") or ""
                articles.append({
                    "title": clean_text(item.get("title", "Untitled Article")),
                    "source": get_domain_name(url),
                    "date": "",
                    "formatted_date": "Recent",
                    "snippet": clean_text(item.get("body", "")),
                    "url": url,
                    "image": None
                })
        else:
            for item in raw_news:
                url = item.get("url") or item.get("href") or ""
                raw_date = item.get("date") or ""
                source_name = item.get("source") or get_domain_name(url)
                
                articles.append({
                    "title": clean_text(item.get("title", "Untitled News")),
                    "source": clean_text(source_name),
                    "date": raw_date,
                    "formatted_date": format_date(raw_date),
                    "snippet": clean_text(item.get("body", "")),
                    "url": url,
                    "image": item.get("image")
                })

    except Exception as e:
        # In case of network errors or rate limit, return an empty dataframe safely
        print(f"[FactShield News Service Error]: {e}")
        return pd.DataFrame(columns=["title", "source", "date", "formatted_date", "snippet", "url", "image"])

    df = pd.DataFrame(articles)
    if df.empty:
        return pd.DataFrame(columns=["title", "source", "date", "formatted_date", "snippet", "url", "image"])
        
    return df
