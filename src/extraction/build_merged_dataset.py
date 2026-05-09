#!/usr/bin/env python3
"""
build_merged_dataset.py

Purpose
-------
Merge cleaned comments with recovered submissions to create a single
analysis-ready dataset for NLP + LLM pipeline.

Output:
- one row per comment
- enriched with submission context

Design:
- deterministic
- fully reproducible
- no in-place ambiguity
"""

import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMMENTS_PATH = ROOT / "data/extraction/wsb_comments_2025_clean_structural.csv"
SUBMISSIONS_PATH = ROOT / "data/extraction/wsb_submissions_2025_recovered.csv"
OUTPUT_PATH = ROOT / "data/extraction/wsb_merged_comments_with_submission.csv"


def clean_submission_text(title, body):
    title = '' if pd.isna(title) else str(title).strip()
    body = '' if pd.isna(body) else str(body).strip()

    bad_body = (body == '') or (body.lower() in ['[removed]', '[deleted]'])

    if bad_body:
        return title
    return f"{title}\n\n{body}"


def main():

    print("[INFO] Loading datasets...")
    comments = pd.read_csv(COMMENTS_PATH, low_memory=False)
    submissions = pd.read_csv(SUBMISSIONS_PATH, low_memory=False)

    # normalize keys
    comments['submission_id'] = comments['submission_id'].astype(str)
    submissions['id'] = submissions['id'].astype(str)

    print("[INFO] Building submission_text...")
    submissions['submission_text'] = submissions.apply(
        lambda row: clean_submission_text(row['title'], row['body_text']),
        axis=1
    )

    # keep only needed submission columns
    submissions_reduced = submissions[[
        'id',
        'submission_text',
        'created_utc'
    ]].rename(columns={
        'id': 'submission_id',
        'created_utc': 'submission_created_utc'
    })

    print("[INFO] Merging...")
    merged = comments.merge(
        submissions_reduced,
        on='submission_id',
        how='left'
    )

    # merge validation
    missing_submission = merged['submission_text'].isna().sum()
    print(f"[INFO] Rows without submission context: {missing_submission}")

    if missing_submission > 5:
        print("[WARNING] More missing submissions than expected")

    # build final text field for LLM
    print("[INFO] Creating message_text...")
    merged['message_text'] = merged['body_text'].astype(str)

    # sort properly
    print("[INFO] Sorting...")
    merged = merged.sort_values(
        ['submission_id', 'created_utc', 'thread_position']
    ).reset_index(drop=True)

    # final column selection (clean)
    final_cols = [
        'id',
        'submission_id',
        'thread_position',
        'created_utc',
        'author',
        'score',
        'message_text',
        'submission_text',
        'matched_tickers',
        'match_count',
        'needs_context_filter'
    ]

    merged_final = merged[final_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_final.to_csv(OUTPUT_PATH, index=False)

    print("[INFO] Saved:", OUTPUT_PATH)
    print("[INFO] Final shape:", merged_final.shape)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()