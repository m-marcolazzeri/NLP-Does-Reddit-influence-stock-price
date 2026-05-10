#!/usr/bin/env python3
"""
01_build_corpus.py  —  STEP 1 of the LDA pipeline
───────────────────────────────────────────────────
Builds and saves all corpus artefacts needed by steps 2 and 3.
Run this once (or whenever preprocessing parameters change).

Inputs:
    data/corpus_building/chunks_lda_v1.csv          training chunks (~6 500 docs)
    config/alias_dictionary_v1.json     alias stopword source

Outputs  (data/topic_modeling/):
    dictionary.gensim       Gensim Dictionary (vocabulary)
    corpus_bow.mm           raw BoW corpus  ← used by LDA
    corpus_bow.mm.index
    bigram_model.pkl        frozen Phrases bigram detector
    trigram_model.pkl       frozen Phrases trigram detector

HPC:
    sbatch run_corpus.sh
"""

from __future__ import annotations

import sys
import json
import pickle
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.topic_modeling.config_lda import (
    BASE_DIR, INPUT_LDA_CHUNKS, OUTPUT_DIR, ALIAS_DICT_PATH,
    ALIAS_TARGET_STOCKS, ALIAS_FIELDS,
    ALLOWED_POS, MIN_TOKEN_LEN, DOMAIN_STOPWORDS,
    BIGRAM_MIN_COUNT, BIGRAM_THRESHOLD,
    TRIGRAM_MIN_COUNT, TRIGRAM_THRESHOLD,
    DICT_NO_BELOW, DICT_NO_ABOVE,
)


# ---------------------------------------------------------------------------
# spaCy loader
# ---------------------------------------------------------------------------

def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("[INFO] Downloading en_core_web_sm...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])


# ---------------------------------------------------------------------------
# Alias stopwords
# ---------------------------------------------------------------------------

def build_alias_stopwords(alias_path: Path, nlp) -> set:
    """
    Extract and lemmatise alias words for ALIAS_TARGET_STOCKS.
    Fields used: exact_tickers, company_names, safe_aliases, risky_aliases.
    notes and context_keywords are intentionally excluded.
    """
    with open(alias_path, encoding="utf-8") as f:
        aliases = json.load(f)

    raw_words: set = set()
    for ticker, data in aliases.items():
        if ticker not in ALIAS_TARGET_STOCKS:
            continue
        for field in ALIAS_FIELDS:
            for entry in data.get(field, []):
                for word in entry.split():
                    raw_words.add(word.lower())

    lemmatised: set = set()
    for word in raw_words:
        for token in nlp(word):
            lemmatised.add(token.lemma_.lower())
        lemmatised.add(word)

    print(f"[INFO] Alias stopwords ({len(lemmatised)}): {sorted(lemmatised)}")
    return lemmatised


# ---------------------------------------------------------------------------
# Tokenisation + lemmatisation
# ---------------------------------------------------------------------------

def preprocess_texts(texts: list[str], nlp, alias_stopwords: set) -> list[list[str]]:
    """
    Tokenise, lemmatise, POS-filter, and remove stopwords.
    Excludes alias stopwords and DOMAIN_STOPWORDS on top of spaCy's list.
    """
    _stop = alias_stopwords | DOMAIN_STOPWORDS
    processed = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        for doc in nlp.pipe(batch, batch_size=batch_size):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if (
                    token.pos_ in ALLOWED_POS
                    and not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                    and len(token.lemma_) >= MIN_TOKEN_LEN
                    and token.lemma_.isalpha()
                    and token.lemma_.lower() not in _stop
                )
            ]
            processed.append(tokens)

        if i % (batch_size * 10) == 0:
            print(f"[INFO]   {min(i + batch_size, len(texts)):,}/{len(texts):,} docs preprocessed")

    return processed


# ---------------------------------------------------------------------------
# Bigrams + Trigrams
# ---------------------------------------------------------------------------

def build_bigrams_and_trigrams(token_lists: list[list[str]]) -> tuple:
    """
    Two chained Gensim Phrases passes: bigrams then trigrams.
    Returns (trigrammed_tokens, bigram_phraser, trigram_phraser).
    """
    from gensim.models import Phrases
    from gensim.models.phrases import Phraser

    print("[INFO] Training bigram detector...")
    bigram_model   = Phrases(token_lists, min_count=BIGRAM_MIN_COUNT, threshold=BIGRAM_THRESHOLD)
    bigram_phraser = Phraser(bigram_model)
    bigrammed      = [bigram_phraser[tokens] for tokens in token_lists]
    n_bi = sum(1 for doc in bigrammed for t in doc if t.count("_") == 1)
    print(f"[INFO] Bigram tokens in corpus: {n_bi:,}")

    print("[INFO] Training trigram detector...")
    trigram_model   = Phrases(bigrammed, min_count=TRIGRAM_MIN_COUNT, threshold=TRIGRAM_THRESHOLD)
    trigram_phraser = Phraser(trigram_model)
    trigrammed      = [trigram_phraser[t] for t in bigrammed]
    n_tri = sum(1 for doc in trigrammed for t in doc if t.count("_") >= 2)
    print(f"[INFO] Trigram tokens in corpus: {n_tri:,}")

    return trigrammed, bigram_phraser, trigram_phraser


# ---------------------------------------------------------------------------
# Dictionary + BoW
# ---------------------------------------------------------------------------

def build_dictionary_and_corpus(token_lists: list[list[str]]):
    """
    Build Gensim Dictionary and raw BoW corpus (LDA input).
    LDA requires raw integer counts; documents are represented as
    bag-of-words vectors after frequency-based vocabulary filtering.
    """
    from gensim.corpora import Dictionary

    print("[INFO] Building dictionary...")
    dictionary = Dictionary(token_lists)
    n_before = len(dictionary)
    dictionary.filter_extremes(no_below=DICT_NO_BELOW, no_above=DICT_NO_ABOVE)
    dictionary.compactify()
    print(f"[INFO] Dictionary: {n_before:,} → {len(dictionary):,} tokens "
          f"(no_below={DICT_NO_BELOW}, no_above={DICT_NO_ABOVE})")

    corpus_bow = [dictionary.doc2bow(tokens) for tokens in token_lists]
    n_empty = sum(1 for doc in corpus_bow if len(doc) == 0)
    if n_empty:
        print(f"[WARN] {n_empty} empty documents after filtering.")

    return dictionary, corpus_bow


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv.field_size_limit(10_000_000)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading training chunks: {INPUT_LDA_CHUNKS.name}")
    lda_df = __import__("pandas").read_csv(INPUT_LDA_CHUNKS, low_memory=False)
    print(f"[INFO] Training chunks: {len(lda_df):,}")

    print("[INFO] Loading spaCy model...")
    nlp = load_spacy()

    print("[INFO] Building alias stopword list...")
    alias_sw = build_alias_stopwords(ALIAS_DICT_PATH, nlp)

    print(f"\n[INFO] Preprocessing {len(lda_df):,} training chunks...")
    texts      = lda_df["chunk_text"].astype(str).tolist()
    token_lists = preprocess_texts(texts, nlp, alias_sw)

    print("\n[INFO] Building bigrams + trigrams...")
    token_lists_tg, bigram_phraser, trigram_phraser = build_bigrams_and_trigrams(token_lists)

    with open(OUTPUT_DIR / "bigram_model.pkl", "wb") as f:
        pickle.dump(bigram_phraser, f)
    print("[INFO] bigram_model.pkl saved.")

    with open(OUTPUT_DIR / "trigram_model.pkl", "wb") as f:
        pickle.dump(trigram_phraser, f)
    print("[INFO] trigram_model.pkl saved.")

    print("\n[INFO] Building dictionary + BoW corpus...")
    dictionary, corpus_bow = build_dictionary_and_corpus(token_lists_tg)

    dictionary.save(str(OUTPUT_DIR / "dictionary.gensim"))
    print("[INFO] dictionary.gensim saved.")

    from gensim.corpora import MmCorpus
    MmCorpus.serialize(str(OUTPUT_DIR / "corpus_bow.mm"), corpus_bow)
    print("[INFO] corpus_bow.mm saved (LDA input).")

    print("\n[INFO] Step 1 complete. Next: sbatch run_search_k.sh")


if __name__ == "__main__":
    main()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        