#!/usr/bin/env python3
"""
clean_text.py

Text cleaning pipeline.

Input:
    data/relevance/relevance_predictions_v1.csv   (239k rows, all comments)

Output:
    data/corpus_building/cleaned_v1.csv                 (relevant comments only, cleaned)

Steps
-----
1.  Filter to predicted_relevance == 1
2.  Convert timestamp → datetime + date
3.  Reddit-specific cleanup (emotes, spoilers, markdown links, bold)
4.  Emoji handling (converted to words for LDA — classical model)
5.  Lowercase
6.  Remove residual URLs
7.  Remove Reddit artifacts (u/user, r/sub, quote lines)
7b. Remove Reddit inline commands/emotes (!\\w+ mid-text)
8.  Keep letters, spaces, $, !, ? — digits removed entirely
9.  Normalize whitespace
10. Remove duplicates
11. Remove comments with fewer than MIN_WORDS words
"""

from __future__ import annotations

import re
import sys
import csv
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH  = BASE_DIR / "data/relevance/relevance_predictions_v1.csv"
OUTPUT_PATH = BASE_DIR / "data/corpus_building/cleaned_v1.csv"

MIN_WORDS = 3  # lowered from 5: captures short but valid comments post-cleaning

# Set to True for classical models (LDA): emojis converted to words.
# Set to False for transformer models (BERT): emojis passed as-is.
USE_EMOJI_CONVERSION = True

OUTPUT_COLUMNS = [
    "id", "submission_id", "thread_position", "created_utc", "date",
    "author", "score", "matched_tickers", "main_stock_or_company",
    "message_text", "clean_text",
]

# ---------------------------------------------------------------------------
# Emoji handler
# ---------------------------------------------------------------------------

if USE_EMOJI_CONVERSION:
    try:
        import emoji as _emoji
        def handle_emojis(text: str) -> str:
            text = _emoji.demojize(text)           # 🚀 → :rocket:
            text = text.replace(":", " ")          # :rocket: → rocket
            text = text.replace("_", " ")          # rocket → rocket (multi-word: thumbs up)
            return text
    except ImportError:
        print("[WARN] 'emoji' package not installed — skipping emoji conversion.")
        def handle_emojis(text: str) -> str:
            return text
else:
    def handle_emojis(text: str) -> str:
        return text


# ---------------------------------------------------------------------------
# Cleaning function
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text)

    # 1. Reddit proprietary emote embeds — no meaningful content, remove entirely
    #    e.g. ![img](emote|t5_2th52|4260)
    text = re.sub(r"!\[img\]\(emote[^)]*\)", " ", text)

    # 2. Spoiler tags — keep the hidden text, remove markers
    #    e.g. >!NVDA to $150!<  →  NVDA to $150
    text = re.sub(r">!(.*?)!<", r"\1", text)

    # 3. Markdown links — keep descriptive text, remove URL and brackets
    #    [some text](https://...) → some text
    #    [this](https://...)      → (generic placeholder, remove entirely)
    def replace_md_link(m: re.Match) -> str:
        link_text = m.group(1).strip()
        return link_text if len(link_text) > 3 else ""
    text = re.sub(r"\[([^\]]*)\]\(https?://[^\)]+\)", replace_md_link, text)

    # 4. Residual URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 5. Bold / italic markdown — keep text, remove markers
    #    **bold** → bold   *italic* → italic
    text = re.sub(r"\*{1,3}([^\*\n]+)\*{1,3}", r"\1", text)

    # 6. Reddit quote lines (lines starting with >)
    text = re.sub(r"(?m)^>+\s?", " ", text)

    # 7. Reddit user / subreddit mentions
    text = re.sub(r"u/\w+", " ", text)
    text = re.sub(r"r/\w+", " ", text)

    # 7b. Reddit inline commands and emotes: !banbet, !remindme, !emote, !guh, etc.
    #     These appear mid-text and were not caught by the pre-filter (which only
    #     matches commands at the start of the message).
    #     Pattern: ! immediately followed by one or more word characters.
    #     Standalone ! (exclamation mark with space after) is preserved.
    text = re.sub(r"!\w+", " ", text)

    # 8. Emoji handling (convert to words or leave as-is)
    text = handle_emojis(text)

    # 9. Lowercase
    text = text.lower()

    # 10. Keep letters, spaces, $, !, ? — digits removed entirely.
    #     Numbers (prices, counts) are too specific to generalize across LDA documents;
    #     the semantic signal comes from words (bullish, puts, dump, etc.), not figures.
    text = re.sub(r"[^a-z\s$!?]", " ", text)

    # 11. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv.field_size_limit(10_000_000)

    print(f"[INFO] Input:  {INPUT_PATH.name}")
    print(f"[INFO] Output: {OUTPUT_PATH.name}")

    print("[INFO] Loading data...")
    df = pd.read_csv(INPUT_PATH, low_memory=False, on_bad_lines="skip")
    print(f"[INFO] Total rows loaded: {len(df):,}")

    # ---------- 1. Filter to relevant comments ----------
    df = df[df["predicted_relevance"] == 1].copy()
    print(f"[INFO] After relevance filter (==1): {len(df):,}")

    # ---------- 2. Convert timestamp ----------
    df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df["date"] = df["created_datetime"].dt.date

    # ---------- 3. Apply cleaning ----------
    print("[INFO] Cleaning text...")
    df["clean_text"] = df["message_text"].apply(clean_text)

    # ---------- 4. Remove duplicates ----------
    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    print(f"[INFO] Duplicates removed: {before - len(df):,}")

    # ---------- 5. Remove short comments ----------
    before = len(df)
    df = df[df["clean_text"].notna()]
    df = df[df["clean_text"].str.split().str.len() >= MIN_WORDS]
    print(f"[INFO] Removed < {MIN_WORDS} words: {before - len(df):,}")

    print(f"[INFO] Final rows: {len(df):,}")

    # ---------- 6. Select and save ----------
    existing_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[existing_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved: {OUTPUT_PATH}  shape={df.shape}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
