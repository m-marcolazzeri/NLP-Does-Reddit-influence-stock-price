#!/usr/bin/env python3
"""
build_remaining_chunks.py

Purpose
-------
Compare thread_chunks_v1.csv (all chunks) against thread_summaries_by_chunk_v1.csv
(processed chunks) and write the difference to thread_chunks_remaining_v1.csv.

Run this after a partial summarizer run to identify what still needs to be processed.

Input
-----
- data/summarization/thread_chunks_v1.csv
- data/summarization/thread_summaries_by_chunk_v1.csv

Output
------
- data/summarization/thread_chunks_remaining_v1.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

ALL_CHUNKS_PATH  = BASE_DIR / "data/summarization/thread_chunks_v1.csv"
DONE_PATH        = BASE_DIR / "data/summarization/thread_summaries_by_chunk_v1.csv"
OUTPUT_PATH      = BASE_DIR / "data/summarization/thread_chunks_remaining_v1.csv"


def main():
    chunks    = pd.read_csv(ALL_CHUNKS_PATH, low_memory=False)
    summaries = pd.read_csv(DONE_PATH, usecols=["submission_id", "chunk_id"], low_memory=False)

    done = set(zip(summaries["submission_id"], summaries["chunk_id"]))
    mask = chunks.apply(lambda r: (r["submission_id"], r["chunk_id"]) in done, axis=1)
    remaining = chunks[~mask].copy()

    print(f"Total chunks:     {len(chunks)}")
    print(f"Already done:     {len(done)}")
    print(f"Remaining:        {len(remaining)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    remaining.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
