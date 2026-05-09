#!/usr/bin/env python3
"""
filter_to_3stocks.py

Purpose
-------
Filter the full 10-stock interim dataset down to the 3 stocks selected
for the main analysis pipeline: NVDA, PLTR, AMD.

Input:
- data/extraction/wsb_comments_2025.csv   (10-stock extraction, ~519k rows — kept as-is)

Output:
- data/extraction/wsb_comments_3stocks.csv  (~248k rows, same schema)

Filter rule
-----------
Keep a row if at least one ticker in `matched_tickers` is in {NVDA, PLTR, AMD}.
`matched_tickers` uses `|` as separator for multi-ticker matches
(e.g. "AMD|NVDA", "NVDA|PLTR|TSLA").

This is the entry point of the 3-stock pipeline. All downstream stages
(cleaning → recovery → merge → chunking → summarization) operate on this
filtered file and overwrite their outputs in place.

The original 10-stock file (wsb_comments_2025.csv) is never modified.
"""

from pathlib import Path
import pandas as pd

TARGET_TICKERS = {"NVDA", "PLTR", "AMD"}

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH  = ROOT / "data/extraction/wsb_comments_2025.csv"
OUTPUT_PATH = ROOT / "data/extraction/wsb_comments_3stocks.csv"


def matches_target(matched_tickers: str) -> bool:
    """Return True if any ticker in the pipe-separated string is a target ticker."""
    return bool(set(str(matched_tickers).split("|")) & TARGET_TICKERS)


def main() -> None:
    print(f"[INFO] Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"[INFO] Rows loaded: {len(df):,}")

    mask = df["matched_tickers"].apply(matches_target)
    df_filtered = df[mask].copy()

    print(f"[INFO] Rows after 3-stock filter: {len(df_filtered):,}  "
          f"({len(df_filtered)/len(df)*100:.1f}% of original)")
    print(f"[INFO] Ticker distribution in filtered dataset:")
    print(df_filtered["matched_tickers"].value_counts().head(20).to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved: {OUTPUT_PATH}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
