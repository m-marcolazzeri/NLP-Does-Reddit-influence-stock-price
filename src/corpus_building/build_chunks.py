#!/usr/bin/env python3
"""
build_chunks.py

Builds two sets of pseudo-documents (chunks) from cleaned Reddit comments.

Input:
    data/corpus_building/cleaned_v1.csv   (195k relevant comments, clean_text column)

Outputs:
    data/corpus_building/chunks_lda_v1.csv        — for LDA *training*
    data/corpus_building/chunks_sentiment_v1.csv  — for LDA *inference* (sentiment scores)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK 1 — LDA training  (chunks_lda_v1.csv)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Population : ALL messages mentioning at least one of {NVDA, AMD, PLTR}
             (including multi-stock messages)

Grouping   : Thread-based.
  - Threads with >= MIN_SIZE messages are chunked internally (groups of
    TARGET_SIZE; short last group absorbed into the preceding one).
  - Threads with < MIN_SIZE messages are pooled together and then chunked
    chronologically across threads.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK 2 — Sentiment inference  (chunks_sentiment_v1.csv)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Population : ONLY single-stock pure messages
             (matched_tickers exactly "NVDA", "AMD", or "PLTR")

Grouping   : Per-stock chronological.
             For each stock independently, sort all messages by created_utc
             and create sequential chunks of TARGET_SIZE.

Chunk size rules (both outputs):
    TARGET_SIZE = 30   — target number of messages per chunk
    MIN_SIZE    = 15   — minimum; if the last group is shorter, it is either
                         absorbed into the previous chunk (same thread) or
                         dropped (cross-thread pool tail).

Columns (both files):
    chunk_id, stock, date_start, date_end, n_messages, chunk_text
"""

from __future__ import annotations

import sys
import csv
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH       = BASE_DIR / "data/corpus_building/cleaned_v1.csv"
OUTPUT_LDA       = BASE_DIR / "data/corpus_building/chunks_lda_v1.csv"
OUTPUT_SENTIMENT = BASE_DIR / "data/corpus_building/chunks_sentiment_v1.csv"

TARGET_STOCKS = {"NVDA", "AMD", "PLTR"}
TARGET_SIZE   = 30
MIN_SIZE      = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: int, stock: str, msgs: pd.DataFrame) -> Dict[str, Any]:
    """Build one output row from a slice of the DataFrame."""
    return {
        "chunk_id":   chunk_id,
        "stock":      stock,
        "date_start": str(msgs["date"].min()),
        "date_end":   str(msgs["date"].max()),
        "n_messages": len(msgs),
        "chunk_text": " [SEP] ".join(msgs["clean_text"].astype(str).tolist()),
    }


def _mentions_target(ticker_str: str) -> bool:
    return bool({t.strip() for t in str(ticker_str).split("|")} & TARGET_STOCKS)


def _dominant_stock(msgs: pd.DataFrame) -> str:
    """Single target ticker if all messages share one, else 'MIXED'."""
    all_tickers: set = set()
    for t in msgs["matched_tickers"].astype(str):
        all_tickers |= {x.strip() for x in t.split("|")} & TARGET_STOCKS
    return next(iter(all_tickers)) if len(all_tickers) == 1 else "MIXED"


def _chunk_sequential(
    df: pd.DataFrame,
    stock_label: str | None,
    chunk_id_start: int,
    absorb_short_tail: bool = True,
) -> tuple[List[Dict[str, Any]], int]:
    """
    Split a sorted DataFrame into sequential chunks of TARGET_SIZE.

    Parameters
    ----------
    df                : sorted messages to chunk
    stock_label       : stock name for every chunk (pass None to infer per chunk)
    chunk_id_start    : first chunk_id to assign
    absorb_short_tail : if True, a tail < MIN_SIZE is merged into the preceding
                        chunk; if False (pool), it is dropped when < MIN_SIZE.
    """
    records: List[Dict[str, Any]] = []
    cid = chunk_id_start
    n   = len(df)
    i   = 0

    while i < n:
        end = i + TARGET_SIZE
        remaining = n - end   # messages left AFTER this slice

        if absorb_short_tail and 0 < remaining < MIN_SIZE:
            # Absorb the short tail into the current chunk
            group = df.iloc[i:]
            i = n
        else:
            group = df.iloc[i:end]
            i = end

        if len(group) < MIN_SIZE:
            # Discard tiny tail (only happens in pool mode)
            break

        label = stock_label if stock_label is not None else _dominant_stock(group)
        records.append(_make_chunk(cid, label, group))
        cid += 1

    return records, cid


# ---------------------------------------------------------------------------
# LDA training chunks  (thread-based + pool for short threads)
# ---------------------------------------------------------------------------

def build_lda_chunks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build chunks for LDA training from all target-stock messages.
    Threads with >= MIN_SIZE messages are chunked internally.
    Shorter threads are pooled and chunked chronologically.
    """
    records: List[Dict[str, Any]] = []
    chunk_id = 0

    # Split threads into "long" (self-contained) and "short" (go to pool)
    thread_sizes = df.groupby("submission_id").size()
    long_threads  = set(thread_sizes[thread_sizes >= MIN_SIZE].index)
    short_threads = set(thread_sizes[thread_sizes <  MIN_SIZE].index)

    print(f"[INFO]   Threads with >= {MIN_SIZE} messages: {len(long_threads):,}")
    print(f"[INFO]   Threads with <  {MIN_SIZE} messages (pooled): {len(short_threads):,}")

    # ── Long threads: chunk per thread ──────────────────────────────────────
    # Sort threads by their earliest message (chronological order of threads)
    thread_start_utc = df[df["submission_id"].isin(long_threads)] \
                         .groupby("submission_id")["created_utc"].min() \
                         .sort_values()

    for submission_id in thread_start_utc.index:
        thread_df = (
            df[df["submission_id"] == submission_id]
            .sort_values("thread_position")
            .reset_index(drop=True)
        )
        stock_label = _dominant_stock(thread_df)
        new_records, chunk_id = _chunk_sequential(
            thread_df, stock_label, chunk_id, absorb_short_tail=True
        )
        records.extend(new_records)

    # ── Short threads: pool and chunk chronologically ───────────────────────
    pool_df = (
        df[df["submission_id"].isin(short_threads)]
        .sort_values("created_utc")
        .reset_index(drop=True)
    )
    print(f"[INFO]   Pool messages: {len(pool_df):,}")
    new_records, chunk_id = _chunk_sequential(
        pool_df, None, chunk_id, absorb_short_tail=False
    )
    records.extend(new_records)

    return records


# ---------------------------------------------------------------------------
# Sentiment inference chunks  (per-stock, per-day)
# ---------------------------------------------------------------------------

def build_sentiment_chunks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build per-stock daily chunks for LDA sentiment inference.
    Only single-stock pure messages are included.

    Grouping: (stock, date) — a chunk never spans two different days.
    Within each day:
      - messages sorted by created_utc
      - split into groups of TARGET_SIZE
      - if the last group would be < MIN_SIZE, absorb into the preceding one
      - days with fewer than TARGET_SIZE messages produce one chunk (no discard)
    """
    records: List[Dict[str, Any]] = []
    chunk_id = 0

    for stock in sorted(TARGET_STOCKS):
        stock_df = df[df["matched_tickers"] == stock].copy()
        stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date

        dates = sorted(stock_df["date"].unique())
        n_chunks_stock = 0

        for date in dates:
            day_df = (
                stock_df[stock_df["date"] == date]
                .sort_values("created_utc")
                .reset_index(drop=True)
            )
            n = len(day_df)
            i = 0

            while i < n:
                end = i + TARGET_SIZE
                remaining = n - end

                if 0 < remaining < MIN_SIZE:
                    # Absorb the short tail into the current chunk
                    group = day_df.iloc[i:]
                    i = n
                else:
                    group = day_df.iloc[i:end]
                    i = end

                # No minimum-size discard: keep every day's messages,
                # even if a single day has only 1–2 messages.
                records.append(_make_chunk(chunk_id, stock, group))
                chunk_id += 1
                n_chunks_stock += 1

        print(f"[INFO]   {stock}: {len(stock_df):,} messages -> {n_chunks_stock} chunks "
              f"across {len(dates)} days")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv.field_size_limit(10_000_000)

    print(f"[INFO] Loading: {INPUT_PATH.name}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["created_utc"]      = pd.to_numeric(df["created_utc"],      errors="coerce")
    df["thread_position"]  = pd.to_numeric(df["thread_position"],  errors="coerce").fillna(1).astype(int)
    print(f"[INFO] Total rows: {len(df):,}")

    # ── LDA training chunks ────────────────────────────────────────────────
    print("\n[INFO] Building LDA training chunks...")
    lda_df = df[df["matched_tickers"].apply(_mentions_target)].copy()
    print(f"[INFO] Messages mentioning >=1 target stock: {len(lda_df):,}")

    lda_records = build_lda_chunks(lda_df)
    lda_out = pd.DataFrame(lda_records).drop(columns=["stock"])

    OUTPUT_LDA.parent.mkdir(parents=True, exist_ok=True)
    lda_out.to_csv(OUTPUT_LDA, index=False)
    print(f"[INFO] LDA chunks saved: {OUTPUT_LDA.name}  shape={lda_out.shape}")
    print(f"[INFO] n_messages: mean={lda_out['n_messages'].mean():.1f}  "
          f"min={lda_out['n_messages'].min()}  max={lda_out['n_messages'].max()}")

    # ── Sentiment inference chunks ─────────────────────────────────────────
    print("\n[INFO] Building sentiment inference chunks (single-stock only)...")
    sentiment_df = df[df["matched_tickers"].isin(TARGET_STOCKS)].copy()
    print(f"[INFO] Single-stock messages total: {len(sentiment_df):,}")

    sentiment_records = build_sentiment_chunks(sentiment_df)
    sent_out = pd.DataFrame(sentiment_records)

    OUTPUT_SENTIMENT.parent.mkdir(parents=True, exist_ok=True)
    sent_out.to_csv(OUTPUT_SENTIMENT, index=False)
    print(f"[INFO] Sentiment chunks saved: {OUTPUT_SENTIMENT.name}  shape={sent_out.shape}")
    print(f"[INFO] n_messages: mean={sent_out['n_messages'].mean():.1f}  "
          f"min={sent_out['n_messages'].min()}  max={sent_out['n_messages'].max()}")
    print(f"[INFO] Stock distribution:\n{sent_out['stock'].value_counts().to_string()}")
    # Verify no chunk spans multiple dates
    cross_date = sent_out[sent_out["date_start"] != sent_out["date_end"]]
    if len(cross_date) == 0:
        print("[INFO] OK - All sentiment chunks are within a single day.")
    else:
        print(f"[WARN] {len(cross_date)} chunks span multiple days — check logic!")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
