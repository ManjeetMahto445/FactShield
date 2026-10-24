"""
FactShield — Fact Checking & Evidence Verification Engine
Author: FactShield Development Team
Description: Searches credible sources, scrapes accessible content, computes
TF-IDF sentence-claim similarity with Scikit-learn, identifies publisher ratings,
and determines an evidence-based assessment.
"""

import re
from typing import Dict, List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import clean_text, split_into_sentences, get_domain_name
from scraper import fetch_article_text

# Resilient DDGS import
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Well-known independent fact-checking publishers
KNOWN_FACT_CHECKERS = {
    "snopes.com": "Snopes",
    "politifact.com": "PolitiFact",
    "factcheck.org": "FactCheck.org",
    "reuters.com": "Reuters Fact Check",
    "apnews.com": "AP Fact Check",
    "bbc.com": "BBC Reality Check",
    "washingtonpost.com": "Washington Post Fact Checker",
    "afp.com": "AFP Fact Check",
    "leadstories.com": "Lead Stories",
    "fullfact.org": "Full Fact",
    "usatoday.com": "USA Today Fact Check",
    "checkyourfact.com": "Check Your Fact",
}

# Stance markers used to detect refutation vs support
CONTRADICTION_PATTERNS = [
    r"\bfalse\b",
    r"\bdebunk(ed|s|ing)?\b",
    r"\bhoax\b",
    r"\bmisleading\b",
    r"\bfake\b",
    r"\bfabricated\b",
    r"\bno evidence\b",
    r"\bunproven\b",
    r"\brefute(d|s)?\b",
    r"\bdisprove(n|d|s)?\b",
    r"\bdenie(d|s)?\b",
    r"\binaccurate\b",
    r"\bunfounded\b",
    r"\bnot true\b",
    r"\bincorrect\b",
    r"\brumor\b",
    r"\bmisinformation\b",
    r"\bdistorted\b",
    r"\bnever happened\b",
    r"\bwrongly claim(ed|s)?\b",
]

SUPPORT_PATTERNS = [
    r"\bconfirm(ed|s|ing)?\b",
    r"\bverif(ied|ies|ying)\b",
    r"\bofficially announced\b",
    r"\breports confirm\b",
    r"\bevidence shows\b",
    r"\bproved that\b",
    r"\bdocumented that\b",
    r"\btrue that\b",
    r"\bstudies show\b",
    r"\bdata indicates\b",
    r"\bfound that\b",
    r"\bannounc(ed|es|ing)?\b",
    r"\breleas(ed|es|ing)?\b",
    r"\bcaptur(ed|es|ing)?\b",
    r"\bdiscover(ed|s|ing)?\b",
    r"\bdemonstrat(ed|es|ing)?\b",
    r"\bobserv(ed|es|ing)?\b",
    r"\breport(s|ed|ing)? that\b",
    r"\baccording to\b",
]


def extract_search_keywords(claim: str, max_words: int = 6) -> str:
    """
    Extracts high-information keywords from a claim, stripping conversational filler
    so search engines return relevant index entries instead of 0 results.
    """
    stop_words = {
        "a", "an", "the", "is", "was", "are", "were", "it", "that", "this",
        "on", "in", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "s", "t", "can", "will", "just", "don", "should", "now", "claim", "claims",
        "says", "said"
    }
    words = re.findall(r"\b[a-zA-Z0-9]+\b", claim)
    meaningful = [w for w in words if w.lower() not in stop_words]
    if not meaningful:
        meaningful = words
    return " ".join(meaningful[:max_words])


def identify_publisher_fact_check(url: str, title: str, text: str) -> Dict[str, Any] | None:
    """
    Checks if an article comes from an established fact-checking organization
    and detects any explicit verdict/rating issued by that publisher.
    
    Returns a dict with publisher details or None if not an established fact-checker.
    """
    domain = get_domain_name(url)
    publisher_name = None
    
    # Match domain with known fact checking organizations
    for known_domain, name in KNOWN_FACT_CHECKERS.items():
        if known_domain in domain:
            publisher_name = name
            break
            
    # Also check if title or URL explicitly says "fact check"
    if not publisher_name:
        if "fact-check" in url.lower() or "fact check" in title.lower():
            publisher_name = f"{domain.capitalize()} (Fact Check)"
        else:
            return None

    # Detect rating words in headline and initial text
    headline_and_intro = f"{title}. {text[:500]}".lower()
    
    detected_rating = "Investigative Fact Check"
    is_refutation = False
    is_affirmation = False
    
    if any(k in headline_and_intro for k in ["pants on fire", "false", "debunked", "hoax", "fake", "incorrect"]):
        detected_rating = "Rated FALSE / Debunked"
        is_refutation = True
    elif any(k in headline_and_intro for k in ["misleading", "mostly false", "partly false", "missing context", "out of context"]):
        detected_rating = "Rated MISLEADING / Partial False"
        is_refutation = True
    elif any(k in headline_and_intro for k in ["mostly true", "correct", "verified", "accurate", "true"]):
        detected_rating = "Rated TRUE / Verified"
        is_affirmation = True
    elif "unproven" in headline_and_intro:
        detected_rating = "Rated UNPROVEN"

    return {
        "publisher": publisher_name,
        "article_title": title,
        "url": url,
        "detected_rating": detected_rating,
        "is_refutation": is_refutation,
        "is_affirmation": is_affirmation,
        "summary": text[:220] + "..." if len(text) > 220 else text
    }


def search_claim_evidence(claim: str, max_sources: int = 6) -> List[Dict[str, Any]]:
    """
    Multi-tiered resilient search strategy:
    1. Targeted query: keywords + 'fact check'
    2. Primary query: keywords
    3. News query: keywords
    
    Gracefully handles DDGS exceptions and deduplicates URLs.
    """
    cleaned_claim = clean_text(claim)
    if not cleaned_claim:
        return []
        
    keywords = extract_search_keywords(cleaned_claim)
    if not keywords:
        keywords = cleaned_claim[:40]

    ddgs = DDGS()
    combined_results = []
    seen_urls = set()

    def add_results(items):
        for item in items:
            url = item.get("href") or item.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined_results.append({
                    "title": clean_text(item.get("title", "")),
                    "url": url,
                    "snippet": clean_text(item.get("body", "")),
                    "source": get_domain_name(url)
                })

    # Tier 1: Search targeted fact-checks
    try:
        r1 = list(ddgs.text(f"{keywords} fact check", max_results=max_sources))
        add_results(r1)
    except Exception as e:
        print(f"[DDGS Tier 1 Info]: {e}")

    # Tier 2: Search broad keyword topic if under max_sources
    if len(combined_results) < max_sources:
        try:
            r2 = list(ddgs.text(keywords, max_results=max_sources))
            add_results(r2)
        except Exception as e:
            print(f"[DDGS Tier 2 Info]: {e}")

    # Tier 3: Search news articles for the topic
    if len(combined_results) < max_sources:
        try:
            r3 = list(ddgs.news(keywords, max_results=max_sources))
            add_results(r3)
        except Exception as e:
            print(f"[DDGS Tier 3 Info]: {e}")

    return combined_results[:max_sources]


def analyze_claim(claim: str, max_sources: int = 5) -> Dict[str, Any]:
    """
    Main Fact-Checking Pipeline:
    1. Validates user input.
    2. Gathers web sources via DuckDuckGo.
    3. Scrapes accessible text using BeautifulSoup.
    4. Identifies established publisher fact-check ratings.
    5. Tokenizes text into sentences and computes TF-IDF Cosine Similarity against the claim.
    6. Identifies stance (support vs contradiction) without conflating similarity with truth.
    7. Synthesizes evidence to assign one of 4 defined assessment outcomes.
    
    Returns structured analysis dictionary.
    """
    if not claim or len(claim.strip()) < 4:
        return {
            "status": "error",
            "message": "Please enter a specific claim to fact-check.",
            "assessment": "Insufficient evidence",
            "badge_type": "insufficient",
            "assessment_details": "Claim string was too short or empty.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "publisher_fact_checks": [],
            "confidence_score": 0.0,
            "sources_analyzed": 0
        }

    raw_sources = search_claim_evidence(claim, max_sources=max_sources)
    
    if not raw_sources:
        return {
            "status": "warning",
            "message": "No online search results were found for this claim.",
            "assessment": "Insufficient evidence",
            "badge_type": "insufficient",
            "assessment_details": "No indexed articles or reports matched the submitted claim.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "publisher_fact_checks": [],
            "confidence_score": 0.0,
            "sources_analyzed": 0
        }

    publisher_fact_checks = []
    candidate_sentences = []
    
    # Process each source
    for src in raw_sources:
        url = src["url"]
        title = src["title"]
        snippet = src["snippet"]
        source_name = src["source"]
        
        # Scrape article body safely with fallback
        fetch_result = fetch_article_text(url, fallback_snippet=snippet)
        full_text = fetch_result["text"]
        
        # Check if source is a known fact check publisher
        fact_check = identify_publisher_fact_check(url, title, full_text)
        if fact_check:
            publisher_fact_checks.append(fact_check)
            
        # Extract candidate sentences
        sentences = split_into_sentences(full_text)
        # Also include title and snippet as candidate pieces
        if title:
            sentences.insert(0, title)
        if snippet and snippet not in full_text:
            sentences.append(snippet)
            
        for sentence in sentences:
            if len(sentence) >= 25:
                candidate_sentences.append({
                    "sentence": sentence,
                    "source": source_name,
                    "url": url,
                    "title": title
                })

    if not candidate_sentences:
        return {
            "status": "warning",
            "message": "Sources were found, but text was inaccessible.",
            "assessment": "Insufficient evidence",
            "badge_type": "insufficient",
            "assessment_details": "Available sources could not be scraped or contained insufficient text.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "publisher_fact_checks": publisher_fact_checks,
            "confidence_score": 0.1,
            "sources_analyzed": len(raw_sources)
        }

    # TF-IDF Cosine Similarity Computation using Scikit-Learn
    sentence_texts = [item["sentence"] for item in candidate_sentences]
    corpus = [claim] + sentence_texts
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        # Claim vector is row 0; candidate sentences are rows 1:
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    except Exception as e:
        print(f"[TF-IDF Computation Error]: {e}")
        similarities = np.zeros(len(candidate_sentences))

    # Pair each sentence with its similarity score
    scored_sentences = []
    for idx, item in enumerate(candidate_sentences):
        score = float(similarities[idx])
        if score >= 0.10:  # Relevance threshold
            scored_sentences.append({
                **item,
                "similarity_score": round(score, 3)
            })

    # Sort sentences by descending relevance
    scored_sentences.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Classify sentences into Supporting, Contradicting, or Neutral
    supporting_evidence = []
    contradicting_evidence = []
    
    seen_texts = set()

    for item in scored_sentences:
        sent_lower = item["sentence"].lower()
        
        # Deduplicate near-identical sentences
        normalized_sample = sent_lower[:50]
        if normalized_sample in seen_texts:
            continue
        seen_texts.add(normalized_sample)

        # Detect contradiction patterns
        has_contradiction = any(re.search(pat, sent_lower) for pat in CONTRADICTION_PATTERNS)
        # Detect support patterns
        has_support = any(re.search(pat, sent_lower) for pat in SUPPORT_PATTERNS)
        
        # Check if the source itself is an explicit fact-check article debunking the claim
        title_lower = item["title"].lower()
        if any(w in title_lower for w in ["false", "debunk", "hoax", "fact check", "fake", "misleading"]):
            has_contradiction = True

        if has_contradiction and not has_support:
            contradicting_evidence.append(item)
        elif has_support and not has_contradiction:
            supporting_evidence.append(item)

    # Limit evidence items to top 5 per stance
    supporting_evidence = supporting_evidence[:5]
    contradicting_evidence = contradicting_evidence[:5]

    # Outcome Assessment Logic (The 4 Required States)
    publisher_refutations = sum(1 for p in publisher_fact_checks if p["is_refutation"])
    publisher_affirmations = sum(1 for p in publisher_fact_checks if p["is_affirmation"])

    num_contra = len(contradicting_evidence) + (publisher_refutations * 2)
    num_supp = len(supporting_evidence) + (publisher_affirmations * 2)
    total_relevant = len(scored_sentences)

    if publisher_refutations > 0 or (num_contra >= 2 and num_supp == 0) or (num_contra >= 3 and num_contra > num_supp * 1.5):
        assessment = "Contradicted by available evidence"
        details = (
            "Multiple reliable sources or professional fact-checkers explicitly refute, "
            "debunk, or identify this claim as false, misleading, or a hoax."
        )
        badge_type = "contradicted"
    elif publisher_affirmations > 0 or (num_supp >= 2 and num_contra == 0) or (num_supp >= 3 and num_supp > num_contra * 1.5):
        assessment = "Supported by available evidence"
        details = (
            "Available verified reports and credible sources provide consistent corroborating "
            "evidence confirming the premise of this claim."
        )
        badge_type = "supported"
    elif num_contra >= 1 and num_supp >= 1:
        assessment = "Conflicting evidence"
        details = (
            "Online sources present divergent statements. Some reports affirm aspects of the claim, "
            "while others dispute, question, or contradict it."
        )
        badge_type = "conflicting"
    else:
        assessment = "Insufficient evidence"
        details = (
            "The available search results do not provide enough conclusive, explicit evidence "
            "to reliably confirm or refute this claim at this time."
        )
        badge_type = "insufficient"

    # Compute transparency confidence score (0.0 to 1.0)
    if assessment == "Insufficient evidence":
        confidence = 0.25 if total_relevant > 0 else 0.05
    else:
        base = 0.52
        base += min(0.30, (num_supp + num_contra) * 0.06)
        if publisher_fact_checks:
            base += 0.14
        confidence = round(min(0.96, base), 2)

    return {
        "status": "success",
        "claim": claim,
        "assessment": assessment,
        "badge_type": badge_type,
        "assessment_details": details,
        "confidence_score": confidence,
        "sources_analyzed": len(raw_sources),
        "publisher_fact_checks": publisher_fact_checks,
        "supporting_evidence": supporting_evidence,
        "contradicting_evidence": contradicting_evidence,
        "total_relevant_quotes": len(scored_sentences)
    }
