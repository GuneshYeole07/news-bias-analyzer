"""
One-time cleanup: remove duplicate articles from the existing dataset.
Deduplicates by URL (primary) and normalized title+source (secondary).
"""
import re
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "data" / "processed" / "articles_with_sentiment.csv"


def normalize_title(title) -> str:
    if pd.isna(title):
        return ""
    return re.sub(r"[^a-z0-9 ]", "", str(title).lower()).strip()


def main():
    if not CSV_PATH.exists():
        print("No dataset found. Nothing to clean.")
        return

    df = pd.read_csv(CSV_PATH)
    before = len(df)
    print(f"Before cleanup: {before} articles")

    # 1. Dedup by URL
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
        print(f"After URL dedup: {len(df)} articles")

    # 2. Dedup by normalized title + source
    df["_norm_title"] = df["title"].apply(normalize_title)
    df["_dedup_key"] = df["_norm_title"] + "||" + df["source"].astype(str).str.lower().str.strip()
    df = df.drop_duplicates(subset=["_dedup_key"], keep="first")
    df = df.drop(columns=["_norm_title", "_dedup_key"])
    print(f"After title+source dedup: {len(df)} articles")

    removed = before - len(df)
    print(f"\nRemoved {removed} duplicates")
    print(f"Final dataset: {len(df)} unique articles")

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved to {CSV_PATH.name}")


if __name__ == "__main__":
    main()
