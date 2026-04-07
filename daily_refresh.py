"""
Daily Refresh Pipeline — News Bias Analyzer
=============================================
Collects fresh articles from NewsAPI, preprocesses them, runs sentiment
analysis, and appends to the existing dataset (deduplicating).

Usage:
    python daily_refresh.py              # run full pipeline once
    python daily_refresh.py --topics 5   # limit number of topics
"""

import io
import os
import sys
import json
import re
import argparse
import requests
import pandas as pd
import nltk
from datetime import datetime, timedelta
from pathlib import Path

# ── ensure NLTK data is present ─────────────────────
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# ── try loading .env ─────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; fall back to env vars

# ── paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
FINAL_CSV = PROC_DIR / "articles_with_sentiment.csv"
REFRESH_JSON = BASE_DIR / "data" / "last_refresh.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── topics ───────────────────────────────────────────
ALL_TOPICS = [
    "technology",
    "politics",
    "sports",
    "artificial intelligence",
    "climate change",
    "economy",
    "health",
    "entertainment",
]

# =====================================================
# HELPERS — Deduplication
# =====================================================
def _normalize_title(title: str) -> str:
    """Lowercase, strip whitespace/punctuation for fuzzy title matching."""
    if pd.isna(title):
        return ""
    return re.sub(r"[^a-z0-9 ]", "", str(title).lower()).strip()


def _load_existing_keys() -> tuple[set, set]:
    """Load existing URLs and normalized title+source pairs from the dataset."""
    existing_urls: set[str] = set()
    existing_titles: set[str] = set()
    if FINAL_CSV.exists():
        try:
            df = pd.read_csv(FINAL_CSV)
            if "url" in df.columns:
                existing_urls = set(df["url"].dropna().astype(str))
            if "title" in df.columns and "source" in df.columns:
                existing_titles = {
                    f"{_normalize_title(t)}||{str(s).lower().strip()}"
                    for t, s in zip(df["title"], df["source"])
                }
        except Exception:
            pass
    return existing_urls, existing_titles


# =====================================================
# STEP 1 — COLLECT
# =====================================================
def collect_articles(api_key: str, topics: list[str], per_topic: int = 100) -> pd.DataFrame:
    """Fetch articles from NewsAPI /v2/everything for each topic.
    Skips articles whose URL or title+source already exist in the dataset."""
    existing_urls, existing_titles = _load_existing_keys()
    if existing_urls:
        print(f"  [i] {len(existing_urls)} known URLs loaded -- duplicates will be skipped")

    all_rows: list[dict] = []
    skipped = 0
    to_date = datetime.now()
    from_date = to_date - timedelta(days=29)  # free tier: 1 month

    for topic in topics:
        print(f"  [*] Fetching '{topic}' ...")
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": topic,
                "apiKey": api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(per_topic, 100),
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
            },
            timeout=30,
        )

        if resp.status_code != 200:
            print(f"    [!] API error {resp.status_code}: {resp.json().get('message', '')}")
            continue

        articles = resp.json().get("articles", [])
        for a in articles:
            url = a.get("url", "")
            if not url or "removed" in url.lower():
                continue

            # Skip if URL already in dataset
            if url in existing_urls:
                skipped += 1
                continue

            title = a.get("title", "")
            source = a.get("source", {}).get("name", "Unknown")
            title_key = f"{_normalize_title(title)}||{source.lower().strip()}"

            # Skip if normalized title+source already in dataset
            if title_key in existing_titles:
                skipped += 1
                continue

            text = ((a.get("description") or "") + " " + (a.get("content") or "")).strip()
            if len(text) < 50:
                continue

            all_rows.append({
                "title": title,
                "text": text,
                "source": source,
                "url": url,
                "published_at": a.get("publishedAt", ""),
                "author": a.get("author", ""),
                "date_collected": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            # Track new keys so we don't add duplicates within this run
            existing_urls.add(url)
            existing_titles.add(title_key)

        print(f"    [OK] {len(articles)} fetched, {len(all_rows)} new so far ({skipped} duplicates skipped)")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    # Final safety dedup on url, then title+source
    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.drop_duplicates(subset=["title", "source"], keep="first")
    return df


# =====================================================
# STEP 2 — PREPROCESS
# =====================================================
_STOP_WORDS = set(stopwords.words("english"))


def _clean(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _remove_stopwords(text: str) -> str:
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in _STOP_WORDS)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add text_clean, text_no_stopwords, word_count columns."""
    df = df.copy()
    df["text_clean"] = df["text"].apply(_clean)
    df["text_no_stopwords"] = df["text_clean"].apply(_remove_stopwords)
    df["word_count"] = df["text_clean"].apply(lambda t: len(t.split()))
    df = df[df["word_count"] > 10]
    return df


# =====================================================
# STEP 3 — SENTIMENT ANALYSIS
# =====================================================
_SIA = SentimentIntensityAnalyzer()


def _sentiment(text: str) -> dict:
    scores = _SIA.polarity_scores(text)
    c = scores["compound"]
    label = "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    return {"compound": c, "pos": scores["pos"], "neg": scores["neg"], "label": label}


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    results = df["text"].apply(_sentiment)
    df["sentiment_compound"] = results.apply(lambda r: r["compound"])
    df["sentiment_label"] = results.apply(lambda r: r["label"])
    df["sentiment_pos"] = results.apply(lambda r: r["pos"])
    df["sentiment_neg"] = results.apply(lambda r: r["neg"])
    return df


# =====================================================
# STEP 4 — MERGE & DEDUPLICATE
# =====================================================
def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """Append new articles to existing CSV, drop duplicates by URL and title."""
    if FINAL_CSV.exists():
        existing = pd.read_csv(FINAL_CSV)
        before = len(existing)
        # Clean existing data of duplicates too
        if "url" in existing.columns:
            existing = existing.drop_duplicates(subset=["url"], keep="first")
        existing = existing.drop_duplicates(subset=["title", "source"], keep="first")
        cleaned = before - len(existing)
        if cleaned > 0:
            print(f"  [i] Removed {cleaned} old duplicates from existing data")
        print(f"  [>] Existing dataset: {len(existing)} articles")
        combined = pd.concat([existing, new_df], ignore_index=True)
        if "url" in combined.columns:
            combined = combined.drop_duplicates(subset=["url"], keep="first")
        combined = combined.drop_duplicates(subset=["title", "source"], keep="first")
    else:
        combined = new_df
    return combined


# =====================================================
# MAIN
# =====================================================
def run(topic_count: int | None = None):
    # Reconfigure stdout to handle Unicode safely on Windows
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        print("[ERROR] NEWSAPI_KEY not found. Set it in .env or as an environment variable.")
        sys.exit(1)

    topics = ALL_TOPICS[:topic_count] if topic_count else ALL_TOPICS

    print("=" * 55)
    print("  News Bias Analyzer -- Daily Refresh")
    print(f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # 1. Collect
    print("\n[1/4] Collecting articles …")
    raw = collect_articles(api_key, topics)
    if raw.empty:
        print("[!] No new articles collected. Exiting.")
        return
    print(f"  [i] Collected {len(raw)} unique articles from {raw['source'].nunique()} sources")

    # Save raw snapshot
    raw.to_csv(RAW_DIR / f"refresh_{datetime.now():%Y%m%d_%H%M%S}.csv", index=False)

    # 2. Preprocess
    print("\n[2/4] Preprocessing …")
    cleaned = preprocess(raw)
    print(f"  [i] {len(cleaned)} articles after cleaning")

    # 3. Sentiment
    print("\n[3/4] Running sentiment analysis …")
    analyzed = add_sentiment(cleaned)
    dist = analyzed["sentiment_label"].value_counts()
    print(f"  (+) Positive: {dist.get('positive', 0)}  (=) Neutral: {dist.get('neutral', 0)}  (-) Negative: {dist.get('negative', 0)}")

    # 4. Merge
    print("\n[4/4] Merging with existing data …")
    final = merge_with_existing(analyzed)
    final.to_csv(FINAL_CSV, index=False)
    print(f"  [>] Final dataset: {len(final)} articles -> {FINAL_CSV.name}")

    # Write refresh timestamp
    meta = {
        "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "articles_added": len(analyzed),
        "total_articles": len(final),
    }
    REFRESH_JSON.write_text(json.dumps(meta, indent=2))
    print(f"\n[OK] Done! Refresh metadata saved to {REFRESH_JSON.name}")

    # Summary
    print("\n" + "=" * 55)
    print(f"  Total articles : {len(final)}")
    print(f"  Unique sources : {final['source'].nunique()}")
    print(f"  New this run   : {len(analyzed)}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News Bias Analyzer daily refresh")
    parser.add_argument("--topics", type=int, default=None, help="Number of topics to fetch (default: all 8)")
    args = parser.parse_args()
    run(topic_count=args.topics)
