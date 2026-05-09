#!/usr/bin/env python3
"""
prepare_threads.py

Purpose
-------
Prepare thread-level data for LLM summarization.

- groups comments by submission_id
- orders by thread_position
- splits threads into chunks
- formats messages with explicit markers
- attaches submission_text

Output:
- data/summarization/thread_chunks_v1.csv

Each row = one chunk of a thread

Design principles:
- reproducible
- deterministic
- no LLM involved
"""

import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data/extraction/wsb_merged_comments_with_submission.csv"
OUTPUT_PATH = ROOT / "data/summarization/thread_chunks_v1.csv"

CHUNK_SIZE = 100


def format_chunk_with_markers(chunk):
    """
    Format messages with explicit structure for LLM.

    Example:
    [MSG 1]
    text...

    [MSG 2]
    text...
    """
    return "\n\n".join(
        [f"[MSG {i+1}]\n{msg}" for i, msg in enumerate(chunk)]
    )


def build_chunks(comments_list, chunk_size):
    """
    Split list into fixed-size chunks
    """
    return [
        comments_list[i:i + chunk_size]
        for i in range(0, len(comments_list), chunk_size)
    ]


def main():

    print("[INFO] Loading dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("[INFO] Sorting by thread...")
    df = df.sort_values(["submission_id", "thread_position"]).copy()

    print("[INFO] Grouping by submission_id...")
    grouped = df.groupby("submission_id")

    rows = []
    total_threads = len(grouped)

    print(f"[INFO] Total threads: {total_threads}")

    for idx, (submission_id, group) in enumerate(grouped):

        submission_text = group["submission_text"].iloc[0]

        comments = group["message_text"].astype(str).tolist()

        chunks = build_chunks(comments, CHUNK_SIZE)

        for chunk_id, chunk in enumerate(chunks):

            rows.append({
                "submission_id": submission_id,
                "chunk_id": chunk_id,
                "n_messages_in_chunk": len(chunk),
                "submission_text": submission_text,
                "chunk_comments": format_chunk_with_markers(chunk)
            })

        if idx % 1000 == 0:
            print(f"[INFO] Processed threads: {idx}/{total_threads}")

    print("[INFO] Creating DataFrame...")
    chunks_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(OUTPUT_PATH, index=False)

    print("[INFO] Saved:", OUTPUT_PATH)
    print("[INFO] Shape:", chunks_df.shape)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()