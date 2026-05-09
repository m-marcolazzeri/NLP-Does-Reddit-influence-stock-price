#!/usr/bin/env python3
"""
recover_parent_submissions.py

Purpose
-------
Extract from the raw Reddit submissions file (.zst) all submissions
whose IDs are required by the cleaned comments dataset.

This is necessary because the filtered submissions dataset does NOT
contain all parent submissions of matched comments.

Input
-----
- data/raw/subreddits25/wallstreetbets_submissions.zst
- data/extraction/required_submission_ids_from_comments_2025.csv

Output
------
- data/extraction/wsb_submissions_2025_recovered.csv

Design principles
-----------------
- streaming (no memory explosion)
- deterministic
- minimal transformation
- reproducible
"""

import json
import zstandard as zstd
import pandas as pd
from pathlib import Path
import io

ROOT = Path(__file__).resolve().parents[2]
INPUT_ZST = ROOT / "data/raw/subreddits25/wallstreetbets_submissions.zst"
INPUT_IDS = ROOT / "data/extraction/required_submission_ids_from_comments_2025.csv"
OUTPUT_PATH = ROOT / "data/extraction/wsb_submissions_2025_recovered.csv"


def main():

    print("[INFO] Loading required submission IDs...")
    ids_df = pd.read_csv(INPUT_IDS)
    required_ids = set(ids_df["submission_id"].astype(str))
    print(f"[INFO] Required IDs: {len(required_ids)}")

    print("[INFO] Starting streaming extraction from .zst...")

    dctx = zstd.ZstdDecompressor()
    recovered = []
    seen = 0
    matched = 0

    with open(INPUT_ZST, "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:

                seen += 1

                try:
                    post = json.loads(line)
                except Exception:
                    continue

                post_id = str(post.get("id"))

                if post_id in required_ids:
                    matched += 1

                    recovered.append({
                        "id": post_id,
                        "created_utc": post.get("created_utc"),
                        "date_utc": None,  # optional, can reconstruct later
                        "source_type": "submission",
                        "subreddit": post.get("subreddit"),
                        "author": post.get("author"),
                        "score": post.get("score"),
                        "title": post.get("title"),
                        "body_text": post.get("selftext"),
                        "raw_text": None,  # can reconstruct later if needed
                        "permalink": post.get("permalink"),
                        "link_id": None,
                        "parent_id": None,
                        "submission_id": post_id,
                        "matched_tickers": None,
                        "matched_terms": None,
                        "match_sources": None,
                        "match_count": None,
                        "is_multi_match": None,
                        "match_confidence": None,
                        "needs_context_filter": None
                    })

                if seen % 1_000_000 == 0:
                    print(f"[INFO] Seen: {seen:,} | Matched: {matched:,}")

    print("[INFO] Extraction finished")
    print(f"[INFO] Total seen: {seen:,}")
    print(f"[INFO] Total matched: {matched:,}")

    df = pd.DataFrame(recovered)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[INFO] Saved to: {OUTPUT_PATH}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()