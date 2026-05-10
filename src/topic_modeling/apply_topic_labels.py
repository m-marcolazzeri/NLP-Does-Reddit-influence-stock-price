#!/usr/bin/env python3
"""
apply_topic_labels.py
─────────────────────
Adds a topic_name column to data/topic_modeling/topic_words_v1.csv using
the human-assigned labels in config/topic_labels_v1.json.

Run this once after filling in config/topic_labels_v1.json.
Re-run any time labels are updated.

Usage:
    python src/topic_modeling/apply_topic_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR    = Path(__file__).resolve().parents[2]
LABELS_PATH = BASE_DIR / "config" / "topic_labels_v1.json"
TW_PATH     = BASE_DIR / "data" / "topic_modeling" / "topic_words_v1.csv"


def topic_label(topic_id: int, labels: dict) -> str:
    """Return 'name - topic N' if a name is set, else 'topic_N'."""
    entry = labels.get(f"topic_{topic_id}", {})
    name = entry.get("name", "").strip() if isinstance(entry, dict) else str(entry).strip()
    if name:
        return f"{name} - topic {topic_id}"
    return f"topic_{topic_id}"


def main() -> None:
    if not LABELS_PATH.exists():
        print(f"[ERROR] Labels file not found: {LABELS_PATH}")
        print("[INFO]  Run 03_train_lda.py first to generate the template.")
        sys.exit(1)

    if not TW_PATH.exists():
        print(f"[ERROR] Topic words file not found: {TW_PATH}")
        sys.exit(1)

    labels = json.loads(LABELS_PATH.read_text())

    # Report which topics have names and which are still empty.
    topic_keys = sorted([k for k in labels if k.startswith("topic_")],
                        key=lambda x: int(x.split("_")[1]))
    empty = [k for k in topic_keys
             if not (labels[k].get("name", "").strip() if isinstance(labels[k], dict)
                     else str(labels[k]).strip())]
    if empty:
        print(f"[WARN] No name set for: {', '.join(empty)} — will use numeric fallback.")

    df = pd.read_csv(TW_PATH)
    df["topic_name"] = df["topic_id"].apply(lambda i: topic_label(i, labels))

    # Reorder: topic_id, topic_name, rank, word, probability
    cols = ["topic_id", "topic_name"] + [c for c in df.columns
                                          if c not in ("topic_id", "topic_name")]
    df = df[cols]
    df.to_csv(TW_PATH, index=False)

    print(f"[INFO] topic_words_v1.csv updated with topic_name column.")
    print(f"[INFO] Labels applied:")
    for k in topic_keys:
        entry = labels[k]
        name = entry.get("name", "").strip() if isinstance(entry, dict) else str(entry).strip()
        print(f"  {k}: {name or '(empty — fallback used)'}")


if __name__ == "__main__":
    main()
