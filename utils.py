"""
FactShield — Utility Functions
Author: FactShield Development Team
Description: Helper functions for text normalization, sentence tokenization, 
date formatting, and domain extraction.
"""

import re
from datetime import datetime
from urllib.parse import urlparse


def clean_text(text: str) -> str:
    """
    Cleans raw web text by removing non-printable characters, 
    extra whitespaces, and normalizing quotes.
    """
    if not text:
        return ""
    # Normalize unicode quotation marks and dashes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    # Replace multiple whitespaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str, min_length: int = 20) -> list[str]:
    """
    Splits a body of text into individual sentences using rule-based regex.
    Protects common abbreviations (e.g., U.S., Dr., etc.) from false splits.
    
    Args:
        text: The source text to split.
        min_length: Minimum character length for a valid sentence.
        
    Returns:
        List of clean sentence strings.
    """
    if not text:
        return []

    cleaned = clean_text(text)
    
    # Temporarily replace periods in common abbreviations
    abbr_pattern = r"\b(Dr|Mr|Mrs|Ms|Prof|Gov|Sen|Rep|Gen|Col|Capt|Lt|St|vs|e\.g|i\.e|U\.S|U\.K|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\."
    protected_text = re.sub(abbr_pattern, r"\1<ABBR_DOT>", cleaned, flags=re.IGNORECASE)
    
    # Split on sentence boundaries: punctuation followed by space and capital letter or quote
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\d])", protected_text)
    
    sentences = []
    for s in raw_sentences:
        restored = s.replace("<ABBR_DOT>", ".").strip()
        if len(restored) >= min_length:
            sentences.append(restored)
            
    return sentences


def format_date(date_str: str) -> str:
    """
    Converts ISO or standard timestamp strings to a clean, readable format.
    Example: '2026-10-23T16:18:00+00:00' -> 'Oct 23, 2026'
    """
    if not date_str:
        return "Recent"
        
    try:
        # Handle ISO strings with timezone
        if "T" in date_str:
            clean_iso = date_str.split("+")[0].split("Z")[0]
            dt = datetime.fromisoformat(clean_iso)
            return dt.strftime("%b %d, %Y")
        # Handle simple YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            return dt.strftime("%b %d, %Y")
    except Exception:
        pass
        
    # Return as-is if already human-readable or couldn't parse
    return date_str[:25]


def truncate_text(text: str, max_chars: int = 180) -> str:
    """
    Safely truncates text to max_chars without breaking mid-word.
    """
    if not text or len(text) <= max_chars:
        return text or ""
        
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."


def get_domain_name(url: str) -> str:
    """
    Extracts the clean hostname/domain from a URL.
    Example: 'https://www.reuters.com/fact-check/...' -> 'reuters.com'
    """
    if not url:
        return "Unknown Source"
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or "Web Source"
    except Exception:
        return "Web Source"
