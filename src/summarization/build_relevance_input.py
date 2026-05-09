#!/usr/bin/env python3
"""
build_relevance_input.py

Joins the merged comment dataset with per-chunk LLM summaries.
Join key: chunk_id = (thread_position - 1) // 100
Join on (submission_id, chunk_id).

TEST_MODE = True  -> uses thread_summaries_by_chunk_TEST.csv, writes pre_relevance_TEST.csv
TEST_MODE = False -> uses thread_summaries_by_chunk_v1.csv,   writes pre_relevance_v1.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

COMMENTS_PATH       = BASE_DIR / "data/extraction/wsb_merged_comments_with_submission.csv"
SUMMARIES_PATH      = BASE_DIR / "data/summarization/thread_summaries_by_chunk_v1.csv"
SUMMARIES_PATH_TEST = BASE_DIR / "data/summarization/thread_summaries_by_chunk_TEST.csv"
OUTPUT_PATH         = BASE_DIR / "data/relevance/pre_relevance_v1.csv"
OUTPUT_PATH_TEST    = BASE_DIR / "data/relevance/pre_relevance_TEST.csv"

TEST_MODE = False

# submission_text is read separately to avoid slow combined CSV parsing
LIGHT_COLS = ["id", "submission_id", "thread_position", "created_utc",
              "author", "score", "matched_tickers", "message_text"]

OUTPUT_COLUMNS = [
    "id", "submission_id", "chunk_id", "thread_position", "created_utc",
    "author", "score", "matched_tickers", "message_text", "submission_text",
    "main_stock_or_company", "LLM_summary",
]

INT_COLUMNS = ["thread_position", "chunk_id", "score", "created_utc"]


def main():
    summaries_path = SUMMARIES_PATH_TEST if TEST_MODE else SUMMARIES_PATH
    output_path    = OUTPUT_PATH_TEST    if TEST_MODE else OUTPUT_PATH

    print(f"[INFO] TEST_MODE = {TEST_MODE}")
    print(f"[INFO] Summaries: {summaries_path.name}")
    print(f"[INFO] Output:    {output_path.name}")

    # Read comments in two passes to avoid slow combined parsing of large text cols
    print("[INFO] Loading comments (pass 1/2 — light columns)...")
    comments = pd.read_csv(COMMENTS_PATH, usecols=LIGHT_COLS, low_memory=False)
    print("[INFO] Loading comments (pass 2/2 — submission_text)...")
    sub_text = pd.read_csv(COMMENTS_PATH, usecols=["id", "submission_text"], low_memory=False)

    print("[INFO] Loading summaries...")
    summaries = pd.read_csv(summaries_path, low_memory=False)
    print(f"[INFO] Comments: {comments.shape}  |  Summaries: {summaries.shape}")

    # Drop null row
    comments = comments.dropna(subset=["id"]).copy()
    sub_text = sub_text.dropna(subset=["id"]).copy()

    if TEST_MODE:
        test_ids = set(summaries["submission_id"].unique())
        comments = comments[comments["submission_id"].isin(test_ids)].copy()
        sub_text = sub_text[sub_text["id"].isin(comments["id"])].copy()
        print(f"[INFO] Comments after TEST filter: {comments.shape}")

    # Compute chunk_id
    comments["chunk_id"] = ((comments["thread_position"] - 1) // 100).astype(int)

    # Join with summaries
    print("[INFO] Joining with summaries...")
    df = comments.merge(
        summaries[["submission_id", "chunk_id", "main_stock_or_company", "LLM_summary"]],
        on=["submission_id", "chunk_id"],
        how="left",
    )

    # Join with submission_text
    df = df.merge(sub_text, on="id", how="left")

    n_missing = df["LLM_summary"].isna().sum()
    if n_missing > 0:
        print(f"[WARN] {n_missing} comments have no matching LLM summary ({n_missing/len(df)*100:.1f}%)")
    else:
        print("[INFO] All comments matched to a summary.")
    print(f"[INFO] Final shape: {df.shape}")

    # Select and order columns
    existing_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[existing_cols]

    # Cast to int
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("[INFO] Saving (may take a minute due to large text fields)...")
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved: {output_path}  shape={df.shape}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
