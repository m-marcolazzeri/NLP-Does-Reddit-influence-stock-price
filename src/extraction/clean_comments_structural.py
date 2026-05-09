
#!/usr/bin/env python3
"""
clean_comments_structural.py

Purpose
-------
Create a cleaned structural copy of the extracted comments dataset.

This script performs only the cleaning steps that are justified before
submission-context recovery:

1. Remove rows with missing submission_id
2. Remove rows with duplicate comment id
3. Remove likely bot / moderator rows
4. Optionally remove exact composite duplicates
   based on (submission_id, created_utc, body_text)
5. Sort by submission_id, then created_utc
6. Export:
   - cleaned comments dataset
   - unique required submission IDs for parent-submission recovery
   - cleaning report

Why only structural cleaning here?
----------------------------------
Because semantic relevance will be handled later, after submission context
has been recovered and attached. At this stage we avoid aggressive filtering.

Repository assumptions
----------------------
Project root:
    Reddit-tech-stocks-NLP/

Expected locations:
    data/extraction/wsb_comments_2025.csv
    data/processed/
    src/extraction/

Recommended usage
-----------------
From project root:

python src/filtering/clean_comments_structural.py

Optional flags:
    --drop-composite-dups true|false
    --input path/to/input.csv
    --output-clean path/to/output.csv
    --output-submission-ids path/to/submission_ids.csv
    --output-report path/to/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/extraction/wsb_comments_3stocks.csv"
DEFAULT_OUTPUT_CLEAN = ROOT / "data/extraction/wsb_comments_2025_clean_structural.csv"
DEFAULT_OUTPUT_SUBMISSION_IDS = ROOT / "data/extraction/required_submission_ids_from_comments_2025.csv"
DEFAULT_OUTPUT_REPORT = ROOT / "data/extraction/reports/wsb_comments_2025_clean_structural_report.json"


BOT_MOD_PATTERNS: List[str] = [
    r"(?<![a-z])mod(?![a-z])",   # whole-word "mod" — avoids "Comodo", "Raymond" etc.
    r"bot",
    r"automoderator",
    r"visualmod",
]


REQUIRED_COLUMNS: List[str] = [
    "id",
    "created_utc",
    "date_utc",
    "source_type",
    "subreddit",
    "author",
    "score",
    "title",
    "body_text",
    "raw_text",
    "permalink",
    "link_id",
    "parent_id",
    "submission_id",
    "matched_tickers",
    "matched_terms",
    "match_sources",
    "match_count",
    "is_multi_match",
    "match_confidence",
    "needs_context_filter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a structurally cleaned copy of wsb_comments_2025.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to input comments CSV.",
    )
    parser.add_argument(
        "--output-clean",
        type=Path,
        default=DEFAULT_OUTPUT_CLEAN,
        help="Path to cleaned comments CSV.",
    )
    parser.add_argument(
        "--output-submission-ids",
        type=Path,
        default=DEFAULT_OUTPUT_SUBMISSION_IDS,
        help="Path to CSV containing unique required submission IDs.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
        help="Path to JSON cleaning report.",
    )
    parser.add_argument(
        "--drop-composite-dups",
        type=str,
        default="true",
        choices=["true", "false"],
        help=(
            "Whether to drop duplicate rows defined by "
            "(submission_id, created_utc, body_text). "
            "Default: true"
        ),
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input comments file is missing required columns: "
            + ", ".join(missing)
        )


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["id"] = out["id"].astype(str).str.strip()
    out["submission_id"] = out["submission_id"].astype(str).str.strip()
    out["author"] = out["author"].fillna("").astype(str).str.strip()

    # created_utc should stay numeric for sorting; coerce if needed
    out["created_utc"] = pd.to_numeric(out["created_utc"], errors="coerce")

    # keep body_text as string for duplicate logic
    out["body_text"] = out["body_text"].fillna("").astype(str)

    return out


def remove_missing_submission_id(df: pd.DataFrame) -> pd.DataFrame:
    # Handle true NaN and stringified missing forms defensively
    bad_values = {"", "nan", "none", "null", "<na>"}
    mask_valid = ~df["submission_id"].str.lower().isin(bad_values)
    return df.loc[mask_valid].copy()


def remove_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df.duplicated(subset=["id"], keep="first")].copy()


def remove_bot_mod_rows(df: pd.DataFrame) -> pd.DataFrame:
    pattern = "|".join(BOT_MOD_PATTERNS)
    mask_keep = ~df["author"].str.contains(pattern, case=False, na=False, regex=True)
    return df.loc[mask_keep].copy()


def remove_composite_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # This is a conservative rule discussed during inspection.
    # Keep first occurrence only.
    return df.loc[
        ~df.duplicated(subset=["submission_id", "created_utc", "body_text"], keep="first")
    ].copy()


def sort_comments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["submission_id", "created_utc", "id"], kind="mergesort").copy()
    out["thread_position"] = out.groupby("submission_id").cumcount() + 1
    return out


def build_required_submission_ids(df: pd.DataFrame) -> pd.DataFrame:
    submission_ids = (
        df["submission_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    submission_ids = submission_ids[submission_ids != ""]
    unique_ids = sorted(submission_ids.unique().tolist())
    return pd.DataFrame({"submission_id": unique_ids})


def collect_report(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    removed_missing_submission_id: int,
    removed_duplicate_ids: int,
    removed_bot_mod: int,
    removed_composite_dups: int,
    composite_dups_dropped: bool,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "input_rows": int(len(raw_df)),
        "output_rows": int(len(cleaned_df)),
        "rows_removed_total": int(len(raw_df) - len(cleaned_df)),
        "removed_missing_submission_id": int(removed_missing_submission_id),
        "removed_duplicate_ids": int(removed_duplicate_ids),
        "removed_bot_mod_rows": int(removed_bot_mod),
        "removed_composite_duplicates": int(removed_composite_dups),
        "composite_duplicates_dropped": bool(composite_dups_dropped),
        "unique_submission_ids_needed": int(cleaned_df["submission_id"].nunique()),
        "unique_comment_ids_after_cleaning": int(cleaned_df["id"].nunique()),
        "columns_in_output": cleaned_df.columns.tolist(),
    }
    return report


def main() -> None:
    args = parse_args()
    drop_composite_dups = args.drop_composite_dups.lower() == "true"

    print(f"[INFO] Reading comments file: {args.input}")
    comments = pd.read_csv(args.input, low_memory=False)
    validate_columns(comments)
    comments = standardize_types(comments)

    raw_n = len(comments)

    # Step 1: remove rows with missing / invalid submission_id
    before = len(comments)
    comments = remove_missing_submission_id(comments)
    removed_missing_submission_id = before - len(comments)
    print(f"[INFO] Removed rows with missing/invalid submission_id: {removed_missing_submission_id}")

    # Step 2: remove duplicate comment ids
    before = len(comments)
    comments = remove_duplicate_ids(comments)
    removed_duplicate_ids = before - len(comments)
    print(f"[INFO] Removed duplicate id rows: {removed_duplicate_ids}")

    # Step 3: remove likely bot / moderator rows
    before = len(comments)
    comments = remove_bot_mod_rows(comments)
    removed_bot_mod = before - len(comments)
    print(f"[INFO] Removed likely bot/mod rows: {removed_bot_mod}")

    # Step 4: optionally remove composite duplicates
    removed_composite_dups = 0
    if drop_composite_dups:
        before = len(comments)
        comments = remove_composite_duplicates(comments)
        removed_composite_dups = before - len(comments)
        print(f"[INFO] Removed composite duplicates: {removed_composite_dups}")
    else:
        print("[INFO] Composite duplicate removal disabled.")

    # Step 5: sort within thread and add thread_position
    comments = sort_comments(comments)
    print("[INFO] Sorted comments by submission_id, created_utc, id and added thread_position.")

    # Step 6: export unique required submission IDs
    required_submission_ids = build_required_submission_ids(comments)

    # Ensure output directories exist
    ensure_parent_dir(args.output_clean)
    ensure_parent_dir(args.output_submission_ids)
    ensure_parent_dir(args.output_report)

    # Step 7: write outputs
    comments.to_csv(args.output_clean, index=False)
    required_submission_ids.to_csv(args.output_submission_ids, index=False)

    report = collect_report(
        raw_df=pd.DataFrame(index=range(raw_n)),
        cleaned_df=comments,
        removed_missing_submission_id=removed_missing_submission_id,
        removed_duplicate_ids=removed_duplicate_ids,
        removed_bot_mod=removed_bot_mod,
        removed_composite_dups=removed_composite_dups,
        composite_dups_dropped=drop_composite_dups,
    )

    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Cleaned comments written to: {args.output_clean}")
    print(f"[INFO] Required submission IDs written to: {args.output_submission_ids}")
    print(f"[INFO] Cleaning report written to: {args.output_report}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
