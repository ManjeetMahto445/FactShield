"""
FactShield — Live News Explorer and Misinformation Checker
College Capstone / Final Year Project
Author: FactShield Development Team
Technology: Python, Streamlit, DuckDuckGo Search, BeautifulSoup4, Pandas, Scikit-learn
"""

import streamlit as st
import pandas as pd
from news_service import fetch_latest_news, NEWS_CATEGORIES
from fact_checker import analyze_claim
from utils import truncate_text

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="FactShield — News Explorer & Misinformation Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Global font and headers */
    h1, h2, h3, h4 {
        color: #f0f6fc;
        font-weight: 600;
    }

    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    
    /* News Card */
    .news-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .news-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .news-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    .news-meta {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 10px;
        display: flex;
        gap: 15px;
    }
    .news-snippet {
        font-size: 0.92rem;
        color: #c9d1d9;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    .source-tag {
        background: #21262d;
        color: #79c0ff;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.8rem;
        border: 1px solid #388bfd33;
    }

    /* Assessment Banner Styles */
    .assessment-box {
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        border-width: 2px;
        border-style: solid;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }
    .assessment-supported {
        background: #0e2d1d;
        border-color: #2ea043;
        color: #7ee787;
    }
    .assessment-contradicted {
        background: #3c1214;
        border-color: #f85149;
        color: #ff7b72;
    }
    .assessment-conflicting {
        background: #332308;
        border-color: #d29922;
        color: #e3b341;
    }
    .assessment-insufficient {
        background: #1c2128;
        border-color: #6e7681;
        color: #c9d1d9;
    }

    /* Publisher Fact Check Card */
    .publisher-card {
        background: #1f242c;
        border-left: 5px solid #a371f7;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .publisher-badge {
        background: #8957e5;
        color: #ffffff;
        font-weight: bold;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }

    /* Evidence Excerpt Card */
    .evidence-card {
        background: #161b22;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        border-left-width: 4px;
        border-left-style: solid;
    }
    .evidence-support {
        border-left-color: #2ea043;
        background: #0f1c15;
    }
    .evidence-contra {
        border-left-color: #f85149;
        background: #241314;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "fact_check_claim" not in st.session_state:
    st.session_state["fact_check_claim"] = ""
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Latest News"


def set_claim_and_navigate(claim_text: str):
    """Helper to pre-fill claim in Fact Checker from a news card."""
    st.session_state["fact_check_claim"] = claim_text
    st.session_state["active_nav"] = "Fact Checker"


# ---------------------------------------------------------
# Sidebar Navigation & Project Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("FactShield")
    st.caption("Live News & Misinformation Verification Engine")
    
    st.markdown("---")
    
    nav_selection = st.radio(
        "Navigation",
        ["Latest News", "Fact Checker", "Project Methodology & Viva Guide"],
        index=["Latest News", "Fact Checker", "Project Methodology & Viva Guide"].index(st.session_state["active_nav"]),
        key="nav_radio"
    )
    # Sync radio choice
    st.session_state["active_nav"] = nav_selection

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.markdown("🟢 **Engine**: Live (No API Key Required)")
    st.markdown("🔍 **Search**: DuckDuckGo API")
    st.markdown("📐 **NLP**: Scikit-Learn TF-IDF Cosine Similarity")
    st.markdown("🌐 **Parser**: BeautifulSoup4 Web Extractor")

    st.markdown("---")
    st.caption("Developed as an open-source college engineering capstone. No commercial or paid AI APIs required.")


# =========================================================
# SECTION 1: LATEST NEWS EXPLORER
# =========================================================
if st.session_state["active_nav"] == "Latest News":
    st.title("📰 Live News Explorer")
    st.markdown("Explore verified live news headlines from global publishers. Click **Investigate in Fact Checker** to examine any headline for misinformation.")

    # Controls Row
    col_cat, col_search, col_limit = st.columns([1.2, 2, 0.8])
    
    with col_cat:
        selected_category = st.selectbox(
            "Category Filter",
            options=list(NEWS_CATEGORIES.keys()),
            index=0
        )
    
    with col_search:
        custom_query = st.text_input(
            "Or Search Specific Topic / Keywords",
            placeholder="e.g. artificial intelligence, renewable energy, space exploration..."
        )
        
    with col_limit:
        max_articles = st.slider("Articles", min_value=4, max_value=20, value=8, step=2)

    # Determine query string
    active_query = custom_query.strip() if custom_query.strip() else NEWS_CATEGORIES[selected_category]

    with st.spinner(f"Fetching latest news on '{active_query}'..."):
        news_df = fetch_latest_news(query=active_query, max_results=max_articles)

    if news_df.empty:
        st.warning("⚠️ No recent news articles were found for this query. Try a different search term.")
    else:
        st.write(f"Showing **{len(news_df)}** recent articles:")

        # Render news articles in responsive grid
        for idx, row in news_df.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="news-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span class="source-tag">{row['source']}</span>
                            <span style="font-size: 0.8rem; color: #8b949e; margin-left: 8px;">📅 {row['formatted_date']}</span>
                        </div>
                    </div>
                    <div class="news-title" style="margin-top: 8px;">{row['title']}</div>
                    <div class="news-snippet">{row['snippet']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action Buttons
                c1, c2, _ = st.columns([1.5, 2, 4])
                with c1:
                    st.link_button("🔗 Read Source Article", row["url"], use_container_width=True)
                with c2:
                    if st.button("🔎 Investigate Headline", key=f"fact_btn_{idx}", use_container_width=True):
                        set_claim_and_navigate(row["title"])
                        st.rerun()
                st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# =========================================================
# SECTION 2: FACT CHECKER
# =========================================================
elif st.session_state["active_nav"] == "Fact Checker":
    st.title("🛡️ Misinformation & Fact Checker")
    st.markdown("Investigate claims using multi-source evidence extraction, publisher fact-check detection, and TF-IDF sentence similarity.")

    # Preset Sample Claims for Quick Demonstration
    st.markdown("##### 💡 Try sample claims for quick viva / demo:")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        if st.button("🎬 Moon landing was staged on movie set", use_container_width=True):
            st.session_state["fact_check_claim"] = "The Apollo 11 moon landing was faked on a movie set"
            st.rerun()
            
    with sample_col2:
        if st.button("🔭 JWST captured early galaxies", use_container_width=True):
            st.session_state["fact_check_claim"] = "James Webb Space Telescope observed earliest known galaxies in deep space"
            st.rerun()
            
    with sample_col3:
        if st.button("⚡ Electric cars catch fire more than gas cars", use_container_width=True):
            st.session_state["fact_check_claim"] = "Electric vehicles catch fire more often than gasoline powered cars"
            st.rerun()

    # Input Form
    with st.form("fact_check_form"):
        user_claim = st.text_area(
            "Enter Claim to Investigate:",
            value=st.session_state.get("fact_check_claim", ""),
            placeholder="Type or paste a claim here (e.g., 'Drinking hot water cures COVID-19', 'NASA found flowing water on Mars')...",
            height=90
        )
        
        col_btn, col_depth = st.columns([1, 1.5])
        with col_btn:
            submit_button = st.form_submit_button("🔍 Run Evidence Investigation", use_container_width=True)
        with col_depth:
            search_depth = st.radio(
                "Search Depth:",
                ["Standard (4 Sources)", "Deep Scan (8 Sources)"],
                horizontal=True,
                index=0
            )

    max_src_count = 4 if "Standard" in search_depth else 8

    # Execution
    if submit_button or (user_claim and st.session_state.get("auto_run", False)):
        st.session_state["auto_run"] = False
        
        if not user_claim.strip():
            st.error("Please enter a valid claim to analyze.")
        else:
            with st.status("🔍 Conducting Evidence Investigation...", expanded=True) as status_box:
                st.write("📡 Querying DuckDuckGo for independent fact-checks and news coverage...")
                st.write("🌐 Scraping accessible web text and cleaning DOM elements with BeautifulSoup...")
                st.write("📐 Vectorizing claim and extracted sentences with Scikit-learn TF-IDF...")
                st.write("⚖️ Classifying stance (Support vs Contradiction) and evaluating publisher ratings...")
                
                result = analyze_claim(user_claim.strip(), max_sources=max_src_count)
                status_box.update(label="✅ Evidence Analysis Complete!", state="complete", expanded=False)

            # Display Error / Warning if needed
            if result.get("status") == "error":
                st.error(result.get("message"))
            else:
                # -------------------------------------------------
                # 1. Main Assessment Banner (Requirement 5)
                # -------------------------------------------------
                assessment = result["assessment"]
                badge_type = result["badge_type"]
                details = result["assessment_details"]
                confidence = result["confidence_score"]

                if badge_type == "supported":
                    box_class = "assessment-supported"
                    icon = "✅"
                elif badge_type == "contradicted":
                    box_class = "assessment-contradicted"
                    icon = "❌"
                elif badge_type == "conflicting":
                    box_class = "assessment-conflicting"
                    icon = "⚠️"
                else:
                    box_class = "assessment-insufficient"
                    icon = "❓"

                st.markdown(f"""
                <div class="assessment-box {box_class}">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85;">FactShield Algorithmic Assessment</span>
                            <h2 style="margin: 4px 0 10px 0; color: inherit;">{icon} {assessment}</h2>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.8rem; font-weight: 700;">{int(confidence * 100)}%</div>
                            <div style="font-size: 0.75rem; opacity: 0.8;">Confidence Metric</div>
                        </div>
                    </div>
                    <p style="margin: 0; font-size: 0.98rem; line-height: 1.5; color: inherit;">{details}</p>
                </div>
                """, unsafe_allow_html=True)

                # Transparency Metric Counters
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Sources Scanned", result["sources_analyzed"])
                with m2:
                    st.metric("Publisher Fact-Checks", len(result["publisher_fact_checks"]))
                with m3:
                    st.metric("Supporting Excerpts", len(result["supporting_evidence"]))
                with m4:
                    st.metric("Contradicting Excerpts", len(result["contradicting_evidence"]))

                # -------------------------------------------------
                # 2. Publisher Fact-Check Ratings (Requirement 6)
                # -------------------------------------------------
                st.markdown("---")
                st.markdown("### 🏛️ Verified Publisher Fact-Checks")
                st.markdown(
                    "*The following ratings were issued directly by recognized professional fact-checking organizations "
                    "(e.g., Snopes, PolitiFact, Reuters Fact Check). They are presented independently from FactShield's text similarity analysis.*"
                )

                if result["publisher_fact_checks"]:
                    for pfc in result["publisher_fact_checks"]:
                        st.markdown(f"""
                        <div class="publisher-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <div>
                                    <strong style="color: #d2a8ff; font-size: 1.05rem;">{pfc['publisher']}</strong>
                                    <span class="publisher-badge" style="margin-left: 10px;">{pfc['detected_rating']}</span>
                                </div>
                                <a href="{pfc['url']}" target="_blank" style="color: #58a6ff; font-size: 0.85rem; text-decoration: none;">View Official Rating ↗</a>
                            </div>
                            <div style="color: #f0f6fc; font-weight: 500; margin-bottom: 6px;">{pfc['article_title']}</div>
                            <div style="color: #8b949e; font-size: 0.88rem;">{pfc['summary']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No dedicated fact-checking agency (e.g. Snopes, PolitiFact) has an explicit record cataloged for this exact query. General news sources are examined below.")

                # -------------------------------------------------
                # 3. Evidence Breakdown: Supporting vs Contradicting
                # -------------------------------------------------
                st.markdown("---")
                st.markdown("### 📑 Contextual Evidence Breakdown")
                st.caption("Sentences extracted from readable source text and ranked by Scikit-learn TF-IDF Cosine Similarity.")

                tab_contra, tab_supp = st.tabs([
                    f"🔴 Contradicting / Refuting Evidence ({len(result['contradicting_evidence'])})",
                    f"🟢 Supporting / Corroborating Evidence ({len(result['supporting_evidence'])})"
                ])

                with tab_contra:
                    if result["contradicting_evidence"]:
                        for item in result["contradicting_evidence"]:
                            st.markdown(f"""
                            <div class="evidence-card evidence-contra">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                    <span style="color: #ff7b72; font-weight: 600; font-size: 0.85rem;">SOURCE: {item['source']}</span>
                                    <span style="color: #8b949e; font-size: 0.8rem;">TF-IDF Similarity: <strong>{item['similarity_score']}</strong></span>
                                </div>
                                <div style="color: #e6edf3; font-style: italic; margin-bottom: 8px;">"{item['sentence']}"</div>
                                <a href="{item['url']}" target="_blank" style="color: #58a6ff; font-size: 0.82rem; text-decoration: none;">🔗 Source: {item['title']}</a>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No direct contradicting excerpts were extracted from available sources.")

                with tab_supp:
                    if result["supporting_evidence"]:
                        for item in result["supporting_evidence"]:
                            st.markdown(f"""
                            <div class="evidence-card evidence-support">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                    <span style="color: #7ee787; font-weight: 600; font-size: 0.85rem;">SOURCE: {item['source']}</span>
                                    <span style="color: #8b949e; font-size: 0.8rem;">TF-IDF Similarity: <strong>{item['similarity_score']}</strong></span>
                                </div>
                                <div style="color: #e6edf3; font-style: italic; margin-bottom: 8px;">"{item['sentence']}"</div>
                                <a href="{item['url']}" target="_blank" style="color: #58a6ff; font-size: 0.82rem; text-decoration: none;">🔗 Source: {item['title']}</a>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No direct supporting excerpts were extracted from available sources.")


# =========================================================
# SECTION 3: PROJECT METHODOLOGY & VIVA GUIDE
# =========================================================
elif st.session_state["active_nav"] == "Project Methodology & Viva Guide":
    st.title("🎓 Project Architecture & Academic Viva Guide")
    st.markdown("Detailed breakdown of FactShield's design, pipeline, algorithm choices, and common college examination questions.")

    st.markdown("### 🏗️ System Architecture Pipeline")
    st.markdown("""
    ```
    +-------------------------------------------------------------+
    |                     FactShield UI                           |
    |          (Streamlit Clean Modern Web Dashboard)             |
    +------------------------------+------------------------------+
                                   |
             +---------------------+---------------------+
             |                                           |
             v                                           v
    [ 📰 Latest News Section ]              [ 🛡️ Fact Checker Section ]
             |                                           |
             v                                           v
      news_service.py                             fact_checker.py
             |                                           |
    +--------v----------+                       +--------v----------+
    | DuckDuckGo Search |                       | DuckDuckGo Search |
    |   (Live DDGS)     |                       | (Tiered Queries)  |
    +--------+----------+                       +--------+----------+
             |                                           |
             v                                           v
    [ Pandas DataFrame ]                            scraper.py
    (Categorized & Rendered)                    (Safe BeautifulSoup4)
                                                         |
                                                +--------v----------+
                                                | Candidate Sentence|
                                                |    Extraction     |
                                                +--------+----------+
                                                         |
                                                +--------v----------+
                                                | Scikit-learn TFIDF|
                                                | Cosine Similarity |
                                                +--------+----------+
                                                         |
                                                +--------v----------+
                                                | Stance Detection  |
                                                | (Support/Contra)  |
                                                +--------+----------+
                                                         |
                                                +--------v----------+
                                                | 4-Tier Assessment |
                                                |   & Confidence    |
                                                +-------------------+
    ```
    """)

    st.markdown("---")
    st.markdown("### 🎯 Frequently Asked Viva Questions & Answers")
    
    with st.expander("Q1: What are the four assessment outcomes supported by FactShield?"):
        st.markdown("""
        FactShield supports four strict assessment outcomes based on empirical evidence:
        1. **Supported by available evidence**: Verified reports and credible sources consistently confirm the claim without refutation.
        2. **Contradicted by available evidence**: Recognized fact-checkers (Snopes, PolitiFact, etc.) or reliable sources explicitly debunk or refute the claim.
        3. **Conflicting evidence**: Sources present contradictory statements (some affirm while others dispute).
        4. **Insufficient evidence**: Too few indexed sources or no conclusive evidence exists to confirm or deny the claim.
        """)

    with st.expander("Q2: Why doesn't FactShield use text similarity alone as proof of truth?"):
        st.markdown("""
        **Crucial Engineering Principle**: High text similarity only indicates that an article is discussing the same topic or claim. 
        For example:
        - *Claim*: "The Earth is flat."
        - *Debunking Sentence*: "The claim that the Earth is flat is a thoroughly debunked myth."
        
        The TF-IDF cosine similarity between the claim and the debunking sentence is above 85% because they share key terms ('Earth', 'flat', 'claim'). 
        If an algorithm relied solely on similarity, it would mistakenly classify the debunking article as 'supporting evidence'! 
        FactShield combines **TF-IDF relevance** with **linguistic stance classification** to distinguish whether the text affirms or refutes the claim.
        """)

    with st.expander("Q3: How does FactShield scrape web articles without crashing on blocks or paywalls?"):
        st.markdown("""
        FactShield implements **Safe Scraping with Fallbacks** (`scraper.py`):
        - Requests have a strict 5-second timeout to prevent application hanging.
        - Common browser headers are supplied to prevent generic 403 Forbidden responses.
        - BeautifulSoup strips scripts, styles, navigation bars, and ads.
        - If an article blocks scraping (e.g., paywalls, Cloudflare bot checks, network drops), FactShield automatically falls back to the search engine snippet provided by DuckDuckGo so evidence is never lost.
        """)

    with st.expander("Q4: Why does FactShield distinguish publisher fact-checks from its own assessment?"):
        st.markdown("""
        Established fact-checking agencies like **Snopes, PolitiFact, and Reuters Fact Check** perform rigorous, human-investigated, journalistic journalism. 
        FactShield clearly separates these verified third-party publisher ratings from its own automated algorithmic TF-IDF assessment to ensure transparency and prevent deceptive authority claims.
        """)
