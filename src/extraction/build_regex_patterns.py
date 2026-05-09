from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Pattern

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def compile_term_pattern(term: str, term_type: str) -> Pattern[str]:
    escaped = re.escape(term)
    if term_type == "cashtag":
        pattern = rf"(?<!\w){escaped}(?!\w)"
    elif term_type == "ticker":
        pattern = rf"(?<![A-Za-z0-9$]){escaped}(?![A-Za-z0-9])"
    else:
        pattern = rf"(?<!\w){escaped}(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE)


def compile_context_pattern(keywords: List[str]) -> Pattern[str] | None:
    clean = [re.escape(normalize_phrase(k)) for k in keywords if normalize_phrase(k)]
    if not clean:
        return None
    joined = "|".join(clean)
    return re.compile(rf"(?<!\w)(?:{joined})(?!\w)", flags=re.IGNORECASE)


def filter_dictionary_to_universe(alias_dict: Dict[str, dict], universe: dict) -> Dict[str, dict]:
    selected = set(universe.get("stocks", []))
    if not selected:
        return alias_dict
    return {ticker: cfg for ticker, cfg in alias_dict.items() if ticker in selected}


def build_patterns(dictionary: Dict[str, dict], active_match_sources: List[str]) -> Dict[str, dict]:
    compiled: Dict[str, dict] = {}
    for ticker, cfg in dictionary.items():
        patterns = []
        if "cashtags" in active_match_sources:
            for term in cfg.get("cashtags", []):
                patterns.append((term, "cashtag", compile_term_pattern(term, "cashtag")))
        if "exact_tickers" in active_match_sources:
            for term in cfg.get("exact_tickers", []):
                patterns.append((term, "ticker", compile_term_pattern(term, "ticker")))
        if "company_names" in active_match_sources:
            for term in cfg.get("company_names", []):
                patterns.append((term, "company_name", compile_term_pattern(term, "company_name")))
        compiled[ticker] = {
            "ticker": ticker,
            "company_name": cfg.get("company_name"),
            "patterns": patterns,
            "requires_context": bool(cfg.get("requires_context", False)),
            "context_pattern": compile_context_pattern(cfg.get("context_keywords", [])),
            "context_keywords": cfg.get("context_keywords", []),
            "notes": cfg.get("notes", ""),
        }
    return compiled


def load_compiled_patterns(
    alias_path: Path | None = None,
    settings_path: Path | None = None,
    universe_path: Path | None = None,
) -> Dict[str, dict]:
    alias_path = alias_path or CONFIG_DIR / "alias_dictionary_v1.json"
    settings_path = settings_path or CONFIG_DIR / "extraction_settings_v1.json"
    universe_path = universe_path or CONFIG_DIR / "stock_universe_v1.json"

    alias_dict = load_json(alias_path)
    settings = load_json(settings_path)
    universe = load_json(universe_path)

    alias_dict = filter_dictionary_to_universe(alias_dict, universe)
    active_sources = settings.get("active_match_sources", ["cashtags", "exact_tickers", "company_names"])
    return build_patterns(alias_dict, active_sources)
