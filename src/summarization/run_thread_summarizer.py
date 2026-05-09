#!/usr/bin/env python3
"""
run_thread_summarizer.py

Purpose
-------
Summarize Reddit thread chunks one by one, producing one summary per chunk.

Input:
- data/summarization/thread_chunks_v1.csv

Output:
- data/summarization/thread_summaries_by_chunk_v1.csv

Logic (v2 — one summary per chunk)
------------------------------------
For each thread, chunks are processed in order (chunk_id 0, 1, 2, ...):

- chunk 0  → build_chunk_summary_prompt(submission_text, chunk_comments)
             → produces summary_0  (saved)
- chunk 1  → build_chunk_summary_with_context_prompt(submission_text,
                                                      summary_0, chunk_comments)
             → produces summary_1  (saved)
- chunk N  → build_chunk_summary_with_context_prompt(submission_text,
                                                      summary_{N-1}, chunk_comments)
             → produces summary_N  (saved)

Every chunk produces one row in the output. The previous chunk's summary is
passed as context only — the output always describes the current chunk, not
the whole thread.

Join key for downstream use
---------------------------
To attach the correct chunk summary to each comment in the merged dataset:

    chunk_id = (thread_position - 1) // 100

Join on (submission_id, chunk_id). thread_position is 1-based.

This version includes:
- decoding only newly generated tokens
- robust extraction of the LAST valid JSON object
- controlled output normalization
- strict fallback behavior
- incremental checkpoint saving (one file per CHECKPOINT_EVERY threads)
- resume support: already-processed submission_ids are skipped on restart
"""

from __future__ import annotations

import sys
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.summarization.prompts import (
    build_chunk_summary_prompt,
    build_chunk_summary_with_context_prompt,
)

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH        = BASE_DIR / "data/summarization/thread_chunks_v1.csv"
OUTPUT_PATH       = BASE_DIR / "data/summarization/thread_summaries_by_chunk_v1.csv"
OUTPUT_PATH_TEST  = BASE_DIR / "data/summarization/thread_summaries_by_chunk_TEST.csv"

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

MAX_THREADS: Optional[int] = None   # None = process all threads
MAX_NEW_TOKENS = 500
CHECKPOINT_EVERY = 20               # save to disk every N threads

# ---------------------------------------------------------------------------
# TEST FILTER — set TEST_MODE = True to run a quick sanity check.
# Selects only threads with >= TEST_MIN_CHUNKS_PER_THREAD chunks, stops after
# TEST_MAX_CHUNKS total chunks have been processed.
# ---------------------------------------------------------------------------
TEST_MODE = False
TEST_MIN_CHUNKS_PER_THREAD = 2   # only threads with 2+ chunks
TEST_MAX_CHUNKS = 10             # stop after this many chunks processed
# ---------------------------------------------------------------------------


ALLOWED_FINANCIAL_ANGLES = {
    "price_action",
    "trading_positions",
    "earnings",
    "valuation",
    "company_news",
    "sector_macro",
    "product_or_company_discussion",
    "mixed",
    "unclear",
}

ALLOWED_CONVERSATION_CHARACTERS = {
    "analytical",
    "speculative",
    "reactive",
    "meme_heavy",
    "mixed",
    "off_topic",
}


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_last_valid_json_object(text: str) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []

    for start_idx in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        in_string = False
        escape = False

        for i in range(start_idx, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            candidates.append(parsed)
                    except Exception:
                        pass
                    break

    if not candidates:
        raise ValueError("No valid JSON object found in model output.")

    return candidates[-1]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_summary(summary: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}

    main_stock          = str(summary.get("main_stock_or_company", "unclear")).strip()
    thread_topic        = str(summary.get("thread_topic", "unclear")).strip()
    financial_angle     = str(summary.get("financial_angle", "unclear")).strip()
    conversation_char   = str(summary.get("conversation_character", "unclear")).strip()
    brief_summary = str(summary.get("brief_summary", "unclear")).strip()

    if financial_angle not in ALLOWED_FINANCIAL_ANGLES:
        financial_angle = "unclear"

    if conversation_char not in ALLOWED_CONVERSATION_CHARACTERS:
        conversation_char = "unclear"

    normalized["main_stock_or_company"]  = main_stock if main_stock else "unclear"
    normalized["thread_topic"]           = thread_topic if thread_topic else "unclear"
    normalized["financial_angle"]        = financial_angle
    normalized["conversation_character"] = conversation_char
    normalized["brief_summary"]          = brief_summary if brief_summary else "unclear"

    return normalized


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_summary(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
) -> Dict[str, str]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    try:
        parsed = extract_last_valid_json_object(decoded)
        return normalize_summary(parsed)
    except Exception:
        return {
            "main_stock_or_company":  "unclear",
            "thread_topic":           "unclear",
            "financial_angle":        "unclear",
            "conversation_character": "unclear",
            "brief_summary":          "unclear",
        }


# ---------------------------------------------------------------------------
# LLM_summary assembly
# ---------------------------------------------------------------------------

def assemble_llm_summary(summary: Dict[str, str]) -> str:
    """
    Combine the four descriptive fields into a single LLM_summary string.

    Format:
        [thread_topic]: <value>
        [financial_angle]: <value>
        [conversation_character]: <value>
        [brief_summary]: <value>
    """
    return (
        f"[thread_topic]: {summary['thread_topic']}\n"
        f"[financial_angle]: {summary['financial_angle']}\n"
        f"[conversation_character]: {summary['conversation_character']}\n"
        f"[brief_summary]: {summary['brief_summary']}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[INFO] Loading chunk dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = df.sort_values(["submission_id", "chunk_id"]).copy()

    thread_ids = df["submission_id"].drop_duplicates().tolist()
    if MAX_THREADS is not None:
        thread_ids = thread_ids[:MAX_THREADS]
        df = df[df["submission_id"].isin(thread_ids)].copy()

    # TEST FILTER — remove before full run
    if TEST_MODE:
        chunks_per_thread = df.groupby("submission_id")["chunk_id"].count()
        multi_chunk_ids = chunks_per_thread[chunks_per_thread >= TEST_MIN_CHUNKS_PER_THREAD].index.tolist()
        thread_ids = [t for t in thread_ids if t in set(multi_chunk_ids)]
        df = df[df["submission_id"].isin(thread_ids)].copy()
        print(f"[TEST] TEST_MODE active — threads with >= {TEST_MIN_CHUNKS_PER_THREAD} chunks: {len(thread_ids)}")
        print(f"[TEST] Will stop after {TEST_MAX_CHUNKS} total chunks processed.")

    print(f"[INFO] Total threads to process: {len(thread_ids)}")

    # In TEST_MODE: use a separate output file, always overwrite, no resume.
    # In production: use resume logic to skip already-processed chunks.
    if TEST_MODE:
        active_output = OUTPUT_PATH_TEST
        already_done: set = set()
        print(f"[TEST] Output → {active_output}  (always overwritten, no resume)")
    else:
        active_output = OUTPUT_PATH
        already_done: set = set()
        done_summaries: dict = {}  # (submission_id, chunk_id) -> LLM_summary
        if active_output.exists():
            existing = pd.read_csv(
                active_output, usecols=["submission_id", "chunk_id", "LLM_summary"]
            )
            already_done = set(zip(existing["submission_id"], existing["chunk_id"]))
            done_summaries = {
                (row["submission_id"], row["chunk_id"]): str(row["LLM_summary"])
                for _, row in existing.iterrows()
            }
            print(f"[INFO] Resuming: {len(already_done)} (submission_id, chunk_id) pairs already in output.")

    print("[INFO] Loading model...")
    import os as _os
    _local_only = _os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=_local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=_local_only,
    )
    model.eval()

    results: List[Dict[str, Any]] = []
    chunks_processed = 0  # TEST counter

    print("[INFO] Running per-chunk summarization...")
    for idx, submission_id in enumerate(thread_ids, start=1):
        thread_df = df[df["submission_id"] == submission_id].sort_values("chunk_id")
        submission_text = str(thread_df["submission_text"].iloc[0])

        previous_context: Optional[str] = None  # LLM_summary text from previous chunk

        for _, row in thread_df.iterrows():
            chunk_id          = int(row["chunk_id"])
            n_messages        = int(row["n_messages_in_chunk"])
            chunk_comments    = str(row["chunk_comments"])

            # Skip if already processed in a previous run (production only)
            if (submission_id, chunk_id) in already_done:
                previous_context = done_summaries.get((submission_id, chunk_id))
                continue

            # Build prompt
            if previous_context is None:
                prompt = build_chunk_summary_prompt(
                    submission_text=submission_text,
                    chunk_comments=chunk_comments,
                )
            else:
                prompt = build_chunk_summary_with_context_prompt(
                    submission_text=submission_text,
                    previous_chunk_summary_json=previous_context,
                    chunk_comments=chunk_comments,
                )

            current_summary = generate_summary(tokenizer, model, prompt)
            llm_summary_text = assemble_llm_summary(current_summary)
            previous_context = llm_summary_text
            chunks_processed += 1

            results.append({
                "submission_id":         submission_id,
                "chunk_id":              chunk_id,
                "n_messages_in_chunk":   n_messages,
                "main_stock_or_company": current_summary["main_stock_or_company"],
                "LLM_summary":           llm_summary_text,
            })

        # TEST FILTER — stop after TEST_MAX_CHUNKS total chunks
        if TEST_MODE and chunks_processed >= TEST_MAX_CHUNKS:
            print(f"[TEST] Reached {chunks_processed} chunks — stopping.")
            break

        # Checkpoint every N threads (production only — in TEST_MODE save at the end)
        if not TEST_MODE and idx % CHECKPOINT_EVERY == 0 and results:
            _save(results, active_output, append=active_output.exists())
            results = []
            print(f"[INFO] Checkpoint saved at thread {idx}/{len(thread_ids)}")

        if idx % 5 == 0:
            print(f"[INFO] Processed threads: {idx}/{len(thread_ids)}")

    # Final save — TEST_MODE always writes fresh (append=False)
    if results:
        _save(results, active_output, append=(not TEST_MODE) and active_output.exists())

    print(f"[INFO] Saved: {active_output}")
    import csv as _csv; _csv.field_size_limit(10_000_000)
    total = pd.read_csv(active_output).shape[0]
    print(f"[INFO] Total rows in output: {total}")
    print("[INFO] Done.")


def _save(rows: List[Dict[str, Any]], path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if append and path.exists():
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)


if __name__ == "__main__":
    main()
