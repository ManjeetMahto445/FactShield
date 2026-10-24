"""
FactShield — Safe Article Scraper
Author: FactShield Development Team
Description: Safe HTTP client and BeautifulSoup HTML extractor for news articles.
Handles network errors, anti-scraping blocks, and gracefully falls back to search snippets.
"""

import requests
from bs4 import BeautifulSoup
from utils import clean_text

# Standard desktop browser user-agent to avoid generic 403 blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Request timeout in seconds
REQUEST_TIMEOUT = 5


def fetch_article_text(url: str, fallback_snippet: str = "") -> dict:
    """
    Safely retrieves the main text of an article from a URL.
    
    If the website blocks the request, times out, or requires a subscription,
    it gracefully falls back to the search result snippet without raising exceptions.
    
    Args:
        url: The web URL of the article.
        fallback_snippet: Snippet from search engine to use if URL is unreadable.
        
    Returns:
        Dictionary with:
          - 'text': Extracted readable text or fallback snippet.
          - 'is_full_article': Boolean indicating whether full article text was retrieved.
          - 'error': None or brief error reason if fallback was triggered.
    """
    if not url or not url.startswith("http"):
        return {
            "text": clean_text(fallback_snippet),
            "is_full_article": False,
            "error": "Invalid URL"
        }

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True
        )
        
        if response.status_code != 200:
            return {
                "text": clean_text(fallback_snippet),
                "is_full_article": False,
                "error": f"HTTP {response.status_code}"
            }
            
        # Parse HTML safely
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove noisy or non-content tags
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "noscript"]):
            element.decompose()
            
        # Strategy 1: Look for semantic <article> or <main> container
        container = soup.find("article") or soup.find("main")
        if not container:
            container = soup.body

        if not container:
            return {
                "text": clean_text(fallback_snippet),
                "is_full_article": False,
                "error": "Empty body"
            }

        # Extract text from paragraph tags
        paragraphs = container.find_all("p")
        extracted_paragraphs = [clean_text(p.get_text()) for p in paragraphs if len(p.get_text().strip()) > 30]
        
        full_text = " ".join(extracted_paragraphs)
        
        # If extracted paragraphs yielded sufficient content (>= 150 chars)
        if len(full_text) >= 150:
            return {
                "text": full_text,
                "is_full_article": True,
                "error": None
            }
        else:
            # If paragraphs were scarce, fallback to text snippet + any scraped snippet
            combined = f"{full_text} {fallback_snippet}".strip()
            return {
                "text": clean_text(combined) if combined else clean_text(fallback_snippet),
                "is_full_article": False,
                "error": "Limited text accessible"
            }

    except requests.exceptions.Timeout:
        return {
            "text": clean_text(fallback_snippet),
            "is_full_article": False,
            "error": "Request timed out"
        }
    except requests.exceptions.RequestException as e:
        return {
            "text": clean_text(fallback_snippet),
            "is_full_article": False,
            "error": "Connection error"
        }
    except Exception as e:
        return {
            "text": clean_text(fallback_snippet),
            "is_full_article": False,
            "error": "Parsing error"
        }
