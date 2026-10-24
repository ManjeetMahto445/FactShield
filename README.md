# 🛡️ FactShield — Live News Explorer and Misinformation Checker

> **College Final Year / Capstone Engineering Project**  
> An open-source, evidence-based web application that aggregates real-time news and investigates claims using multi-source web evidence and text similarity algorithms without relying on paid AI models or commercial APIs.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Architecture & Workflow](#-architecture--workflow)
4. [Technology Stack](#-technology-stack)
5. [The 4 Assessment Outcomes](#-the-4-assessment-outcomes)
6. [Why Text Similarity ≠ Truth (Academic Principle)](#-why-text-similarity--truth-academic-principle)
7. [Installation & Local Execution](#-installation--local-execution)
8. [Project File Structure](#-project-file-structure)
9. [Viva Voce & College Defense Guide](#-viva-voce--college-defense-guide)
10. [License & Ethics](#-license--ethics)

---

## 🌟 Project Overview

In today's digital information ecosystem, false narratives, altered claims, and misleading headlines propagate rapidly. Traditional fact-checking is either completely manual or reliant on closed, expensive proprietary AI APIs.

**FactShield** provides a transparent, zero-cost, automated solution:
1. **Live News Explorer**: Pulls real-time headlines across multiple topical domains (Technology, Science, Business, Health, World) via DuckDuckGo.
2. **Misinformation Checker**: Allows users to submit or transfer any claim, searches across authoritative web sources, parses accessible article contents with BeautifulSoup, extracts candidate sentences, calculates relevance via **Scikit-learn TF-IDF Cosine Similarity**, and applies linguistic stance analysis to assign one of four structured outcomes.
3. **Publisher Ratings Distinction**: Transparently isolates and highlights official ratings from accredited fact-checkers (Snopes, PolitiFact, Reuters Fact Check, AP Fact Check) apart from FactShield's own evidence assessment.

---

## 🚀 Key Features

* **Zero Paid APIs / Free to Run**: Uses the open DuckDuckGo search interface (`ddgs`) — no OpenAI, Anthropic, or paid API keys required.
* **Dual Investigation Mechanism**:
  * Scrapes real-time reporting from news outlets.
  * Direct discovery of established third-party fact-checking investigations.
* **Strict Evidence Attribution**: Never hallucinates articles, dates, quotations, or sources. Every quote links directly to its live origin.
* **Fault-Tolerant Web Scraper**: Gracefully handles network timeouts, 403 Forbidden blocks, and paywalls with seamless fallback to verified search engine snippets.
* **Modern Streamlit Dashboard**: Responsive, clean dark-themed UI with custom CSS, visual status banners, metric cards, and one-click demo claim buttons.

---

## 🏗️ Architecture & Workflow

```text
                               +------------------------------------+
                               |          FactShield UI             |
                               |    (Streamlit Web Dashboard)       |
                               +-----------------+------------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
            [ 📰 Latest News Explorer ]                         [ 🛡️ Fact Checker ]
                       |                                                   |
               news_service.py                                      fact_checker.py
                       |                                                   |
             +---------v---------+                               +---------v---------+
             | DuckDuckGo Search |                               | Multi-Tier Search |
             |   (Live DDGS)     |                               | (Targeted+Keyword)|
             +---------+---------+                               +---------+---------+
                       |                                                   |
                       v                                                   v
              [ Pandas DataFrame ]                                     scraper.py
              (Categorized News)                                   (Safe BeautifulSoup4)
                                                                           |
                                                                 +---------v---------+
                                                                 | Sentence Extraction|
                                                                 |   & Cleaning      |
                                                                 +---------+---------+
                                                                           |
                                                                 +---------v---------+
                                                                 | Scikit-learn TFIDF|
                                                                 | Cosine Similarity |
                                                                 +---------+---------+
                                                                           |
                                                                 +---------v---------+
                                                                 | Stance & Publisher|
                                                                 | Rating Analysis   |
                                                                 +---------+---------+
                                                                           |
                                                                 +---------v---------+
                                                                 | 4-Tier Assessment |
                                                                 |  Confidence Score |
                                                                 +-------------------+
```

---

## 💻 Technology Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Standard language for AI, Data Science, and NLP capstones. |
| **Web UI** | Streamlit | Rapid, modern, reactive Python dashboard with zero front-end overhead. |
| **Search Engine** | DuckDuckGo (`ddgs`) | Free, uncapped, privacy-preserving web search requiring no API keys. |
| **Web Parser** | BeautifulSoup4 | Robust HTML document parsing and boilerplate stripping. |
| **Data Processing**| Pandas | Structured tabular handling and filtering of article metadata. |
| **NLP & Similarity**| Scikit-learn (`TfidfVectorizer`) | Computes term frequency-inverse document frequency and cosine similarity vectors. |
| **HTTP Client** | Requests | Lightweight client with custom browser headers and strict timeouts. |

---

## 🎯 The 4 Assessment Outcomes

FactShield categorizes every evaluated claim into one of four strictly defined states:

1. 🟢 **Supported by available evidence**
   * Corroborating reports from verified primary sources confirm the core premise of the claim.
2. 🔴 **Contradicted by available evidence**
   * Certified fact-checkers (Snopes, PolitiFact, Reuters, AP) or credible reporting explicitly debunk, refute, or prove the claim false.
3. 🟡 **Conflicting evidence**
   * Multiple credible sources present contradictory statements (some affirm while others question or dispute).
4. ⚪ **Insufficient evidence**
   * Available search indexes contain too few relevant documents or inconclusive accessible text to substantiate a definitive conclusion.

---

## 🧠 Why Text Similarity ≠ Truth (Academic Principle)

A common mistake in beginner student projects is assuming that high cosine similarity proves a claim is true.

> **Example**:  
> - **Claim**: *"The Earth is flat."*  
> - **Source Sentence**: *"The claim that the Earth is flat has been thoroughly debunked by modern astrophysics."*  

Because both texts share terms like *Earth*, *flat*, and *claim*, **TF-IDF cosine similarity will be very high (> 0.70)**! 
If an algorithm treats similarity as truth, it will mistakenly declare that the debunking article *proves* the Earth is flat.

### How FactShield Solves This:
FactShield uses TF-IDF similarity purely as a **Relevance Filter** to locate sentences addressing the topic. Once candidate sentences are isolated, FactShield's **Stance Classifier** inspects semantic polarity markers:
* **Refutation Patterns**: `debunked`, `false`, `hoax`, `no evidence that`, `fake`, `fabricated`, `unproven`, `denied`.
* **Affirmation Patterns**: `confirmed that`, `officially announced`, `verified that`, `evidence shows`, `documented that`.

This guarantees that debunking articles are correctly mapped to **Contradicting Evidence**.

---

## 📦 Installation & Local Execution

### 1. Prerequisites
Ensure you have **Python 3.10 or newer** installed. Check in your terminal:
```bash
python --version
```

### 2. Clone or Navigate to the Project
```bash
cd FactShield
```

### 3. (Recommended) Create a Virtual Environment
```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Streamlit will automatically launch the local dashboard in your browser at:
`http://localhost:8501`

---

## 📁 Project File Structure

```text
FactShield/
├── app.py              # Main Streamlit dashboard and UI styling
├── news_service.py     # Real-time news retrieval and category aggregation via DDGS
├── fact_checker.py     # Core evidence analysis, TF-IDF ranking, stance detection
├── scraper.py          # Safe HTTP client and BeautifulSoup HTML extractor with fallbacks
├── utils.py            # Sentence tokenization, date formatting, and text helpers
├── requirements.txt    # Frozen Python dependency list
└── README.md           # Documentation, architecture, and viva defense guide
```

---

## 🎓 Viva Voce & College Defense Guide

### Q1: What is TF-IDF and why was it chosen over complex LLMs?
**Answer**: TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a word is to a sentence within a collection of documents. It was chosen because:
1. It runs locally with near-zero latency and zero memory overhead.
2. It requires no costly API tokens or network dependencies on external proprietary models.
3. It provides mathematical transparency (exact cosine similarity vectors), which is fully auditable.

### Q2: How does the system handle paywalled or blocked news sites?
**Answer**: `scraper.py` implements a graceful degradation strategy:
- Requests have a strict 5-second timeout and fake user-agent headers.
- If a site returns a 403 Forbidden, 404, or times out, the system catches the exception and falls back to the clean search snippet provided by DuckDuckGo. This prevents application crashes and guarantees evidence is retained.

### Q3: How are publisher ratings distinguished from FactShield's own scores?
**Answer**: Professional fact-checking organizations (Snopes, PolitiFact, AP, Reuters Fact Check) employ human investigative journalists. FactShield identifies these publishers by domain and displays their verdicts in a dedicated purple card with direct citation links, keeping them visually and functionally distinct from FactShield's algorithmic TF-IDF stance score.

---

## ⚖️ License & Ethics

* FactShield is created strictly for educational and academic research purposes.
* All news headlines and article excerpts remain the intellectual property of their original publishers.
* FactShield adheres to safe search practices and does not store private user queries.
