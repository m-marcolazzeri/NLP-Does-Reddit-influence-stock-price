#!/usr/bin/env python3
"""
03_train_lda.py  —  STEP 3 of the LDA pipeline
────────────────────────────────────────────────
Final LDA training + theta inference on sentiment chunks.
Loads the pre-built corpus (Step 1) and trains the model with K_FINAL
(set by the human in config_lda.py after inspecting Step 2 output).

This step is fast (~2 h on HPC) and can be re-run freely to iterate on:
  - K_FINAL        : number of topics
  - LDA_PASSES     : more passes → better convergence
  - ALPHA / ETA    : Dirichlet priors (see config_lda.py comments)
  - TOPIC_TOP_N    : number of words saved per topic (no retraining needed)

Inputs  (data/topic_modeling/ — produced by Step 1):
    corpus_bow.mm           raw BoW corpus (LDA training input)
    dictionary.gensim       vocabulary
    bigram_model.pkl        bigram phraser  (used for inference)
    trigram_model.pkl       trigram phraser (used for inference)

    data/corpus_building/chunks_sentiment_v1.csv  per-stock daily chunks for inference

Outputs (data/topic_modeling/):
    model/lda_k{K}/         trained LDA model files
    topic_words_v1.csv      top-N words + probability per topic
    results_v1.csv          theta distributions per sentiment chunk

HPC:
    sbatch run_train_lda.sh
"""

from __future__ import annotations

import sys
import csv
import re
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.topic_modeling.config_lda import (
    BASE_DIR, INPUT_SENTIMENT_CHUNKS, OUTPUT_DIR, ALIAS_DICT_PATH,
    ALIAS_TARGET_STOCKS, ALIAS_FIELDS,
    ALLOWED_POS, MIN_TOKEN_LEN, DOMAIN_STOPWORDS,
    K_FINAL,
    LDA_PASSES, WORKERS, ALPHA, ETA, RANDOM_STATE,
    TOPIC_TOP_N, TOPIC_DIVERSITY_N,
)


# ---------------------------------------------------------------------------
# Helpers (shared with 01_build_corpus.py — kept here for self-containedness)
# ---------------------------------------------------------------------------

def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                       check=True)
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def build_alias_stopwords(alias_path: Path, nlp) -> set:
    import json
    with open(alias_path, encoding="utf-8") as f:
        aliases = json.load(f)
    raw: set = set()
    for ticker, data in aliases.items():
        if ticker not in ALIAS_TARGET_STOCKS:
            continue
        for field in ALIAS_FIELDS:
            for entry in data.get(field, []):
                for word in entry.split():
                    raw.add(word.lower())
    lemmatised: set = set()
    for word in raw:
        for token in nlp(word):
            lemmatised.add(token.lemma_.lower())
        lemmatised.add(word)
    return lemmatised


def preprocess_texts(texts: list[str], nlp, alias_stopwords: set) -> list[list[str]]:
    _stop = alias_stopwords | DOMAIN_STOPWORDS
    processed = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        for doc in nlp.pipe(texts[i: i + batch_size], batch_size=batch_size):
            tokens = [
                token.lemma_.lower() for token in doc
                if (token.pos_ in ALLOWED_POS
                    and not token.is_stop and not token.is_punct
                    and not token.is_space
                    and len(token.lemma_) >= MIN_TOKEN_LEN
                    and token.lemma_.isalpha()
                    and token.lemma_.lower() not in _stop)
            ]
            processed.append(tokens)
    return processed


# ---------------------------------------------------------------------------
# Topic diversity (PUW)
# ---------------------------------------------------------------------------

def topic_diversity(model, num_topics: int, topn: int = TOPIC_DIVERSITY_N) -> float:
    sep = re.compile(r"0\.[0-9]+\*")
    all_words = []
    for _, s in model.print_topics(num_topics=num_topics, num_words=topn):
        all_words.extend(re.sub(sep, "", s).replace('"', '').split(' + ')[:topn])
    return round(len(set(all_words)) / (topn * num_topics), 4)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lda(corpus_bow: list, dictionary, k: int):
    from gensim.models import LdaMulticore
    print(f"[INFO] Training LDA: K={k}, passes={LDA_PASSES}, "
          f"alpha={ALPHA}, eta={ETA}...")
    return LdaMulticore(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=k,
        passes=LDA_PASSES,
        workers=WORKERS,
        alpha=ALPHA,
        eta=ETA,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Topic words
# ---------------------------------------------------------------------------

def extract_topic_words(model, k: int) -> pd.DataFrame:
    rows = []
    for tid in range(k):
        for rank, (word, prob) in enumerate(model.show_topic(tid, topn=TOPIC_TOP_N), 1):
            rows.append({"topic_id": tid, "rank": rank,
                         "word": word, "probability": round(prob, 6)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_theta(
    model,
    dictionary,
    bigram_phraser,
    trigram_phraser,
    nlp,
    sentiment_df: pd.DataFrame,
    k: int,
    alias_stopwords: set,
) -> pd.DataFrame:
    """
    Apply identical preprocessing pipeline to sentiment chunks, then infer θ.
    Uses raw BoW (no TF-IDF) — consistent with training.
    """
    texts = sentiment_df["chunk_text"].astype(str).tolist()
    print(f"[INFO] Preprocessing {len(texts):,} sentiment chunks...")
    tokens    = preprocess_texts(texts, nlp, alias_stopwords)
    tokens_bg = [bigram_phraser[t]  for t in tokens]
    tokens_tg = [trigram_phraser[t] for t in tokens_bg]
    bow_sent  = [dictionary.doc2bow(t) for t in tokens_tg]

    print("[INFO] Inferring θ distributions...")
    theta_rows = []
    for bow_doc in bow_sent:
        dist = model.get_document_topics(bow_doc, minimum_probability=0.0)
        row  = {f"topic_{tid}": round(float(p), 6) for tid, p in dist}
        for t in range(k):
            row.setdefault(f"topic_{t}", 0.0)
        theta_rows.append(row)

    meta = sentiment_df[
        ["chunk_id", "stock", "date_start", "date_end", "n_messages"]
    ].reset_index(drop=True)
    return pd.concat([meta, pd.DataFrame(theta_rows)], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv.field_size_limit(10_000_000)

    # ── Load corpus artefacts from Step 1 ───────────────────────────────────
    print("[INFO] Loading corpus artefacts from Step 1...")
    from gensim.corpora import Dictionary, MmCorpus

    for p in [OUTPUT_DIR / "dictionary.gensim",
              OUTPUT_DIR / "corpus_bow.mm",
              OUTPUT_DIR / "bigram_model.pkl",
              OUTPUT_DIR / "trigram_model.pkl"]:
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Run Step 1 first.")

    dictionary = Dictionary.load(str(OUTPUT_DIR / "dictionary.gensim"))
    corpus_bow = list(MmCorpus(str(OUTPUT_DIR / "corpus_bow.mm")))

    with open(OUTPUT_DIR / "bigram_model.pkl",  "rb") as f:
        bigram_phraser  = pickle.load(f)
    with open(OUTPUT_DIR / "trigram_model.pkl", "rb") as f:
        trigram_phraser = pickle.load(f)

    print(f"[INFO] Dictionary: {len(dictionary):,} tokens | "
          f"Corpus: {len(corpus_bow):,} docs")

    # ── Determine K ─────────────────────────────────────────────────────────
    if K_FINAL is None:
        raise ValueError("Set K_FINAL in config_lda.py after inspecting coherence plots.")
    k = K_FINAL
    print(f"[INFO] Using K_FINAL = {k} (set in config_lda.py)")

    # ── Train ────────────────────────────────────────────────────────────────
    model = train_lda(corpus_bow, dictionary, k)

    model_dir  = OUTPUT_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"lda_k{k}"
    model.save(str(model_path))
    print(f"[INFO] Model saved → {model_path}")

    log_perp = model.log_perplexity(corpus_bow)
    print(f"[INFO] Log-perplexity: {log_perp:.4f}  |  "
          f"Perplexity: {np.exp(-log_perp):.2f}")

    # ── Topic words ──────────────────────────────────────────────────────────
    topic_words_df = extract_topic_words(model, k)
    tw_path = OUTPUT_DIR / "topic_words_v1.csv"
    topic_words_df.to_csv(tw_path, index=False)

    div = topic_diversity(model, k)
    print(f"\n[INFO] Topic Diversity (PUW, top-{TOPIC_DIVERSITY_N}): {div:.4f}")
    print(f"[INFO] Topic words saved → {tw_path.name}")

    print(f"\n[INFO] Top-10 words per topic (K={k}):")
    for tid in range(k):
        words = topic_words_df[topic_words_df["topic_id"] == tid].head(10)["word"].tolist()
        print(f"  Topic {tid:2d}: {', '.join(words)}")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading sentiment chunks: {INPUT_SENTIMENT_CHUNKS.name}")
    sent_df = pd.read_csv(INPUT_SENTIMENT_CHUNKS, low_memory=False)
    print(f"[INFO] Sentiment chunks: {len(sent_df):,}")

    print("[INFO] Loading spaCy model...")
    nlp = load_spacy()

    print("[INFO] Building alias stopwords...")
    alias_sw = build_alias_stopwords(ALIAS_DICT_PATH, nlp)

    results_df = infer_theta(model, dictionary, bigram_phraser, trigram_phraser,
                             nlp, sent_df, k, alias_sw)

    results_path = OUTPUT_DIR / "results_v1.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[INFO] results_v1.csv saved → shape={results_df.shape}")

    topic_cols = [c for c in results_df.columns if c.startswith("topic_")]
    print("\n[INFO] Mean θ per stock:")
    print(results_df.groupby("stock")[topic_cols].mean().round(3).to_string())

    print("\n[INFO] Step 3 complete.")
    print("[INFO] → Open notebooks/06_lda_inspection.ipynb to check topics")
    print("[INFO] → If topics need refinement: adjust config_lda.py → sbatch run_train_lda.sh")
    print("[INFO] → When satisfied: open notebooks/07_topic_labeling.ipynb")


if __name__ == "__main__":
    main()
