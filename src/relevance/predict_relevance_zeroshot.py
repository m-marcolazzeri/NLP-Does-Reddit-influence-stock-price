#!/usr/bin/env python3
"""
predict_relevance_zeroshot.py

Zero-shot relevance classifier for Reddit stock comments.
Model: facebook/bart-large-mnli

Two modes
---------
EVAL_MODE = True
    Input : data/relevance/rows_labeled_by_hand_3000.xlsx  (3000 labeled rows)
    Output: data/relevance/relevance_eval_results.csv
    Also prints classification metrics vs ground-truth labels.

EVAL_MODE = False
    Input : data/relevance/pre_relevance_v1.csv  (full ~239k-row dataset)
    Output: data/relevance/relevance_predictions_v1.csv

Label mapping
-------------
 1  Relevant      — model picks HYPOTHESIS_RELEVANT with confidence >= threshold
 0  Borderline    — max confidence on either hypothesis < CONFIDENCE_THRESHOLD
-1  Not relevant  — model picks HYPOTHESIS_NOT_RELEVANT with confidence >= threshold

Hypothesis strings define the relevance criteria directly in the source code below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_NAME = "facebook/bart-large-mnli"
BATCH_SIZE = 32

# Asymmetric confidence thresholds.
# A higher bar for "relevant" reduces false positives.
# A lower bar for "not-relevant" ensures short/meme comments get caught.
THRESHOLD_RELEVANT     = 0.75   # s_rel must exceed this to assign label  1
THRESHOLD_NOT_RELEVANT = 0.50   # s_nrel must exceed this to assign label -1
# If neither threshold is met → label 0 (borderline/uncertain)

# Minimum number of words a comment must have to be sent to the model.
# Comments shorter than this get label -1 directly (too short to classify reliably).
MIN_WORDS = 5

# Regex pattern for WSB bot commands and formulaic messages that carry no
# standalone analytical value (e.g. !banbet, !remindme, !remind).
# These are assigned label -1 without going through the model.
import re as _re
BOT_COMMAND_PATTERN = _re.compile(
    r"^\s*!(?:banbet|remindme|remind|ban|bet)\b",
    flags=_re.IGNORECASE,
)

EVAL_MODE = False

# MAX_ROWS: if set to an integer, only the first N rows of the production input
# are processed. Useful for a quick sanity check before the full HPC run.
# Set to None to process all rows.
MAX_ROWS: int | None = None

INPUT_EVAL  = BASE_DIR / "data/relevance/rows_labeled_by_hand_3000.xlsx"
INPUT_PROD  = BASE_DIR / "data/relevance/pre_relevance_v1.csv"
OUTPUT_EVAL = BASE_DIR / "data/relevance/relevance_eval_results.csv"
OUTPUT_PROD = BASE_DIR / "data/relevance/relevance_predictions_v1.csv"

# ---------------------------------------------------------------------------
# Hypothesis strings — relevance criteria
# ---------------------------------------------------------------------------

HYPOTHESIS_RELEVANT = (
    "This comment expresses stock price expectations, trading positions such as "
    "calls or puts, financial interpretation of earnings or valuation, market "
    "reactions to news, or sector and macro implications affecting the stock."
)

HYPOTHESIS_NOT_RELEVANT = (
    "This comment is a meme, joke, hype, insult, or spam, or discusses products "
    "and services without financial implications, or mentions a stock ticker "
    "without adding any meaningful financial information."
)

CANDIDATE_LABELS = [HYPOTHESIS_RELEVANT, HYPOTHESIS_NOT_RELEVANT]

# ---------------------------------------------------------------------------
# Text assembly
# ---------------------------------------------------------------------------

MAX_CHARS = 3000  # ~900 tokens; leaves room for the two hypotheses within 1024


def build_input_text(row: pd.Series) -> str:
    """
    Combine message_text with context.
    Priority: LLM_summary (production) > submission_text (eval fallback).
    Truncates to MAX_CHARS to respect the model's 1024-token limit.
    """
    msg = str(row.get("message_text", "")).strip()

    llm_summary  = row.get("LLM_summary", "")
    sub_text     = row.get("submission_text", "")

    if pd.notna(llm_summary) and str(llm_summary).strip():
        ctx       = str(llm_summary).strip()
        ctx_label = "THREAD_CONTEXT"
    elif pd.notna(sub_text) and str(sub_text).strip():
        ctx       = str(sub_text).strip()
        ctx_label = "POST_CONTEXT"
    else:
        ctx       = ""
        ctx_label = ""

    combined = f"COMMENT: {msg}\n{ctx_label}: {ctx}" if ctx else f"COMMENT: {msg}"
    return combined[:MAX_CHARS]


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def assign_label(result: dict) -> tuple[int, float]:
    """
    Returns (label, confidence) using asymmetric thresholds.
      1  if s_rel  >= THRESHOLD_RELEVANT      (high bar — reduces false positives)
     -1  if s_nrel >= THRESHOLD_NOT_RELEVANT  (lower bar — catches memes/short comments)
      0  if neither threshold is met          (borderline/uncertain)

    Priority: relevant is checked first; if both thresholds are met (rare),
    the higher-scoring hypothesis wins.
    """
    scores = dict(zip(result["labels"], result["scores"]))
    s_rel  = scores[HYPOTHESIS_RELEVANT]
    s_nrel = scores[HYPOTHESIS_NOT_RELEVANT]

    if s_rel >= THRESHOLD_RELEVANT and s_rel >= s_nrel:
        return 1, s_rel
    if s_nrel >= THRESHOLD_NOT_RELEVANT:
        return -1, s_nrel
    return 0, max(s_rel, s_nrel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    input_path  = INPUT_EVAL  if EVAL_MODE else INPUT_PROD
    output_path = OUTPUT_EVAL if EVAL_MODE else OUTPUT_PROD

    print(f"[INFO] EVAL_MODE          = {EVAL_MODE}")
    print(f"[INFO] Model              = {MODEL_NAME}")
    print(f"[INFO] Threshold relevant = {THRESHOLD_RELEVANT} | not-relevant = {THRESHOLD_NOT_RELEVANT}")
    print(f"[INFO] Input              = {input_path.name}")
    print(f"[INFO] Output             = {output_path.name}")
    if not EVAL_MODE and MAX_ROWS is not None:
        print(f"[INFO] MAX_ROWS           = {MAX_ROWS}  (sanity-check mode)")

    # --- Load ---------------------------------------------------------------
    if input_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(input_path)
    else:
        nrows = MAX_ROWS if (not EVAL_MODE and MAX_ROWS is not None) else None
        df = pd.read_csv(input_path, low_memory=False, nrows=nrows)
    print(f"[INFO] Loaded {len(df):,} rows.")

    # --- Pre-filters (skip model, assign label directly) --------------------
    msg = df["message_text"].fillna("").astype(str)

    # 1) Too short
    too_short   = msg.str.split().str.len() < MIN_WORDS
    # 2) WSB bot commands (!banbet, !remindme, etc.)
    is_bot_cmd  = msg.str.contains(BOT_COMMAND_PATTERN)

    skip_mask = too_short | is_bot_cmd

    n_short   = too_short.sum()
    n_bot     = is_bot_cmd.sum()
    n_skip    = skip_mask.sum()
    if n_short:
        print(f"[INFO] {n_short} comments below MIN_WORDS={MIN_WORDS} → label -1 (skipped).")
    if n_bot:
        print(f"[INFO] {n_bot} bot-command comments (!banbet etc.)  → label -1 (skipped).")

    df_model = df[~skip_mask].copy()
    df_skip  = df[skip_mask].copy()
    df_skip["predicted_relevance"]  = -1
    df_skip["relevance_confidence"] = 0.0

    # --- Build texts --------------------------------------------------------
    print(f"[INFO] Building input texts for {len(df_model):,} comments...")
    texts = [build_input_text(row) for _, row in df_model.iterrows()]

    n_trunc = sum(len(t) == MAX_CHARS for t in texts)
    if n_trunc:
        print(f"[WARN] {n_trunc} texts truncated to {MAX_CHARS} chars "
              f"({n_trunc / len(texts) * 100:.1f}%).")

    # --- Load model ---------------------------------------------------------
    import os as _os
    _local_only = _os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    device = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] Device = {'GPU' if device == 0 else 'CPU'}")
    print(f"[INFO] local_files_only = {_local_only}")
    print("[INFO] Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=_local_only)
    print("[INFO] Loading model...")
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, local_files_only=_local_only
    )
    clf = pipeline(
        "zero-shot-classification",
        model=_model,
        tokenizer=_tokenizer,
        device=device,
        truncation=True,
    )

    # --- Inference ----------------------------------------------------------
    print(f"[INFO] Running inference (batch_size={BATCH_SIZE})...")
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        out   = clf(batch, candidate_labels=CANDIDATE_LABELS,
                    hypothesis_template="{}", multi_label=False)
        results.extend(out)
        if i % (BATCH_SIZE * 10) == 0:
            print(f"[INFO]   {min(i + BATCH_SIZE, len(texts)):,}/{len(texts):,}")

    # --- Assign labels ------------------------------------------------------
    labels, confidences = zip(*[assign_label(r) for r in results])
    df_model["predicted_relevance"] = labels
    df_model["relevance_confidence"] = np.round(confidences, 4)

    # Recombine with pre-filtered rows and restore original order
    df = pd.concat([df_model, df_skip], ignore_index=True)
    df = df.sort_index()

    dist = pd.Series(labels).value_counts().sort_index()
    print("\n[INFO] Predicted label distribution:")
    for lbl, cnt in dist.items():
        print(f"  {lbl:+d} : {cnt:>6,}  ({cnt / len(df) * 100:.1f}%)")

    # --- Evaluation metrics (EVAL_MODE only) --------------------------------
    if EVAL_MODE and "Relevance" in df.columns:
        from sklearn.metrics import classification_report, confusion_matrix

        gt   = df["Relevance"].astype(int)
        pred = pd.Series(labels)

        # For metrics: map 0 → predicted as "uncertain" — exclude from report
        # or map to -1 as conservative choice; we show both.
        pred_no_zero = pred.replace(0, -1)

        print("\n[INFO] Classification report (borderline mapped to -1):")
        print(classification_report(gt, pred_no_zero,
                                    labels=[1, -1],
                                    target_names=["Relevant (1)", "Not relevant (-1)"],
                                    zero_division=0))

        cm = confusion_matrix(gt, pred_no_zero, labels=[1, -1])
        print("Confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm,
                           index=["True 1", "True -1"],
                           columns=["Pred 1", "Pred -1"]))

        n_zero = (pred == 0).sum()
        print(f"\n[INFO] Borderline (0): {n_zero} rows ({n_zero / len(df) * 100:.1f}%)")

    # --- Save ---------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Saved → {output_path}  shape={df.shape}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
