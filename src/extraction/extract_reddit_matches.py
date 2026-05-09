from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import zstandard as zstd

# Ensure src/extraction/ is on sys.path so this script works regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_regex_patterns import load_compiled_patterns

LOGGER = logging.getLogger("extract_reddit_matches")

OUTPUT_COLUMNS = [
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


@dataclass
class MatchRecord:
    ticker: str
    term: str
    term_type: str


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def utc_date_string(created_utc: int | float | None) -> str:
    if created_utc is None:
        return ""
    try:
        dt = datetime.fromtimestamp(int(created_utc), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return ""
    return dt.strftime("%Y-%m-%d")


def sanitize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_source_text(row: dict, source_type: str) -> tuple[str, str, str]:
    if source_type == "submission":
        title = sanitize_text(str(row.get("title", "") or ""))
        selftext = sanitize_text(str(row.get("selftext", "") or ""))
        pieces = [p for p in [title, selftext] if p and p.lower() not in {"[removed]", "[deleted]"}]
        body = selftext
        raw_text = " \n ".join(pieces).strip()
        return title, body, raw_text
    body = sanitize_text(str(row.get("body", "") or ""))
    if body.lower() in {"[removed]", "[deleted]"}:
        body = ""
    return "", body, body


def context_satisfied(text: str, context_pattern: re.Pattern[str] | None) -> bool:
    if context_pattern is None:
        return True
    return bool(context_pattern.search(text))


def collect_matches(text: str, compiled_patterns: Dict[str, dict]) -> List[MatchRecord]:
    matches: List[MatchRecord] = []
    for ticker, cfg in compiled_patterns.items():
        local_matches: List[MatchRecord] = []
        for term, term_type, pattern in cfg["patterns"]:
            if pattern.search(text):
                local_matches.append(MatchRecord(ticker=ticker, term=term, term_type=term_type))
        if not local_matches:
            continue
        if cfg["requires_context"] and not context_satisfied(text, cfg.get("context_pattern")):
            continue
        matches.extend(local_matches)
    return matches


def deduplicate_matches(matches: List[MatchRecord]) -> List[MatchRecord]:
    seen = set()
    unique: List[MatchRecord] = []
    for m in matches:
        key = (m.ticker, m.term.lower(), m.term_type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return unique


def compute_confidence(matches: List[MatchRecord], compiled_patterns: Dict[str, dict]) -> str:
    if not matches:
        return "none"
    term_types = {m.term_type for m in matches}
    tickers = {m.ticker for m in matches}
    if len(tickers) > 1:
        return "multi_stock"
    if "cashtag" in term_types or "ticker" in term_types:
        return "high"
    only_ticker = next(iter(tickers))
    if compiled_patterns[only_ticker].get("requires_context"):
        return "medium"
    return "high"


def iter_zst_jsonl(path: Path) -> Iterable[dict]:
    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_reader:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.debug("Skipping malformed JSON line")
                    continue


def filter_by_period(created_utc: int | float | None, start_date: str, end_date: str) -> bool:
    if created_utc is None:
        return False
    try:
        dt = datetime.fromtimestamp(int(created_utc), tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return False
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    return start <= dt <= end


def build_permalink(row: dict, source_type: str) -> str:
    permalink = row.get("permalink")
    if permalink:
        return str(permalink)
    rid = row.get("id", "")
    if not rid:
        return ""
    if source_type == "submission":
        return f"https://www.reddit.com/comments/{rid}"
    parent_submission_id = str(row.get("link_id", "")).replace("t3_", "")
    if parent_submission_id:
        return f"https://www.reddit.com/comments/{parent_submission_id}/_/{rid}"
    return ""


def process_file(
    input_path: Path,
    output_path: Path,
    source_type: str,
    start_date: str,
    end_date: str,
    compiled_patterns: Dict[str, dict],
    subreddit_name: str,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"seen": 0, "written": 0, "matched": 0, "skipped_subreddit": 0, "skipped_period": 0}

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in iter_zst_jsonl(input_path):
            stats["seen"] += 1
            subreddit = str(row.get("subreddit", "") or "")
            if subreddit.lower() != subreddit_name.lower():
                stats["skipped_subreddit"] += 1
                continue

            created_utc = row.get("created_utc")
            if not filter_by_period(created_utc, start_date, end_date):
                stats["skipped_period"] += 1
                continue

            title, body_text, raw_text = get_source_text(row, source_type)
            if not raw_text:
                continue

            matches = deduplicate_matches(collect_matches(raw_text, compiled_patterns))
            if not matches:
                continue

            stats["matched"] += 1
            tickers = sorted({m.ticker for m in matches})
            confidence = compute_confidence(matches, compiled_patterns)

            link_id = str(row.get("link_id", "") or "")
            parent_id = str(row.get("parent_id", "") or "")
            submission_id = link_id.replace("t3_", "") if source_type == "comment" else str(row.get("id", "") or "")

            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "created_utc": created_utc,
                    "date_utc": utc_date_string(created_utc),
                    "source_type": source_type,
                    "subreddit": subreddit,
                    "author": row.get("author", ""),
                    "score": row.get("score", ""),
                    "title": title,
                    "body_text": body_text,
                    "raw_text": raw_text,
                    "permalink": build_permalink(row, source_type),
                    "link_id": link_id,
                    "parent_id": parent_id,
                    "submission_id": submission_id,
                    "matched_tickers": "|".join(tickers),
                    "matched_terms": "|".join(sorted({m.term for m in matches})),
                    "match_sources": "|".join(sorted({m.term_type for m in matches})),
                    "match_count": len(tickers),
                    "is_multi_match": int(len(tickers) > 1),
                    "match_confidence": confidence,
                    "needs_context_filter": int(any(compiled_patterns[t]["requires_context"] for t in tickers)),
                }
            )
            stats["written"] += 1

            if stats["written"] % 5000 == 0:
                f.flush()
                LOGGER.info("Written %s matched rows so far...", stats["written"])

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract stock-matched rows from Reddit .zst dumps.")
    parser.add_argument("--input", type=Path, required=True, help="Path to .zst input file")
    parser.add_argument("--output", type=Path, required=True, help="Path to output CSV")
    parser.add_argument("--source-type", choices=["comment", "submission"], required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--subreddit", type=str, default="wallstreetbets")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(args.verbose)
    compiled_patterns = load_compiled_patterns()
    stats = process_file(
        input_path=args.input,
        output_path=args.output,
        source_type=args.source_type,
        start_date=args.start_date,
        end_date=args.end_date,
        compiled_patterns=compiled_patterns,
        subreddit_name=args.subreddit,
    )
    LOGGER.info("Finished. Stats: %s", stats)


if __name__ == "__main__":
    main()
