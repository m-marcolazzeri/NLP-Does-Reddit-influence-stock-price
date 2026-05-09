#!/usr/bin/env python3
"""
02_search_k.py  —  STEP 2 of the LDA pipeline
───────────────────────────────────────────────
Coherence-based K search. Loads the pre-built corpus from disk (Step 1),
trains one LDA model per K in K_RANGE, and saves scores + plot.

After this step:
  1. Download coherence_scores.csv and coherence_plot.png from the HPC.
  2. Open notebooks/06_lda_inspection.ipynb locally.
  3. Choose K based on the coherence and diversity plots.
  4. Set K_FINAL in config_lda.py.
  5. Run Step 3: sbatch run_train_lda.sh

Inputs  (data/topic_modeling/ — produced by Step 1):
    corpus_bow.mm       raw BoW corpus
    dictionary.gensim   vocabulary

Outputs (data/topic_modeling/):
    coherence_scores.csv    UMass · C_v · NPMI · Diversity for each K
    coherence_plot.png      visualisation of the scores

HPC:
    sbatch run_search_k.sh
"""

from __future__ import annotations

import sys
import csv
import re
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.topic_modeling.config_lda import (
    OUTPUT_DIR,
    K_RANGE, K_SEARCH_PASSES,
    WORKERS, ALPHA, ETA, RANDOM_STATE,
    TOPIC_DIVERSITY_N,
)


# ---------------------------------------------------------------------------
# Topic diversity (PUW — Proportion of Unique Words)
# ---------------------------------------------------------------------------

def topic_diversity(model, num_topics: int, topn: int = TOPIC_DIVERSITY_N) -> float:
    topic_sep = re.compile(r"0\.[0-9]+\*")
    all_words = []
    for _, topic_str in model.print_topics(num_topics=num_topics, num_words=topn):
        words = re.sub(topic_sep, "", topic_str).replace('"', '').split(' + ')
        all_words.extend(words[:topn])
    return round(len(set(all_words)) / (topn * num_topics), 4)


# ---------------------------------------------------------------------------
# K search
# ---------------------------------------------------------------------------

def search_k(corpus_bow: list, dictionary, token_lists: list) -> pd.DataFrame:
    """
    Train LDA on corpus_bow for each K in K_RANGE (K_SEARCH_PASSES passes).
    Returns DataFrame with coherence metrics and topic diversity per K.
    Coherence metrics are computed on raw token lists (standard practice).
    """
    from gensim.models import LdaMulticore
    from gensim.models.coherencemodel import CoherenceModel

    rows = []
    for k in K_RANGE:
        print(f"[INFO]   Training K={k} ({K_SEARCH_PASSES} passes)...")
        model = LdaMulticore(
            corpus=corpus_bow,
            id2word=dictionary,
            num_topics=k,
            passes=K_SEARCH_PASSES,
            workers=WORKERS,
            alpha=ALPHA,
            eta=ETA,
            random_state=RANDOM_STATE,
        )

        score_umass = CoherenceModel(
            model=model, corpus=corpus_bow,
            dictionary=dictionary, coherence="u_mass"
        ).get_coherence()

        score_cv = CoherenceModel(
            model=model, texts=token_lists,
            dictionary=dictionary, coherence="c_v"
        ).get_coherence()

        score_npmi = CoherenceModel(
            model=model, texts=token_lists,
            dictionary=dictionary, coherence="c_npmi"
        ).get_coherence()

        score_div = topic_diversity(model, k)

        print(f"[INFO]   K={k:2d}  u_mass={score_umass:+.4f}  "
              f"c_v={score_cv:.4f}  c_npmi={score_npmi:+.4f}  diversity={score_div:.4f}")

        rows.append({
            "K":               k,
            "coherence_umass": round(score_umass, 4),
            "coherence_cv":    round(score_cv,    4),
            "coherence_npmi":  round(score_npmi,  4),
            "topic_diversity": score_div,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def save_coherence_plot(scores: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    metrics = [
        ("coherence_umass", "UMass (↑ better)",   "steelblue"),
        ("coherence_cv",    "C_v   (↑ better)",   "darkorange"),
        ("coherence_npmi",  "NPMI  (↑ better)",   "green"),
        ("topic_diversity", "Diversity PUW (↑ better)", "purple"),
    ]

    for ax, (col, title, color) in zip(axes, metrics):
        scores.plot.line(x="K", y=col, ax=ax, marker="o", color=color, legend=False)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(scores["K"])
        ax.set_xlabel("K (num_topics)")

    best_cv_k = int(scores.loc[scores["coherence_cv"].idxmax(), "K"])
    axes[1].axvline(best_cv_k, color="red", linestyle="--", alpha=0.6,
                    label=f"best C_v @ K={best_cv_k}")
    axes[1].legend(fontsize=9)

    fig.suptitle("Coherence & Diversity by K  —  choose K_FINAL in config_lda.py",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Coherence plot saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv.field_size_limit(10_000_000)

    # Load pre-built corpus from Step 1
    print("[INFO] Loading corpus from disk (Step 1 output)...")
    from gensim.corpora import Dictionary, MmCorpus

    dict_path = OUTPUT_DIR / "dictionary.gensim"
    bow_path  = OUTPUT_DIR / "corpus_bow.mm"
    if not dict_path.exists() or not bow_path.exists():
        raise FileNotFoundError(
            "Dictionary or BoW corpus not found. Run Step 1 first: sbatch run_corpus.sh"
        )

    dictionary = Dictionary.load(str(dict_path))
    corpus_bow = list(MmCorpus(str(bow_path)))
    print(f"[INFO] Dictionary: {len(dictionary):,} tokens | Corpus: {len(corpus_bow):,} docs")

    # Reconstruct token lists needed for c_v and c_npmi coherence
    # (these metrics require raw texts, not just the BoW corpus)
    # Note: dictionary.id2token is NOT populated after Dictionary.load() —
    # invert token2id (always populated) instead.
    print("[INFO] Reconstructing token lists from BoW for coherence calculation...")
    id2word = {v: k for k, v in dictionary.token2id.items()}
    token_lists = [[id2word[wid] for wid, cnt in doc for _ in range(cnt)] for doc in corpus_bow]

    # Search K
    print(f"\n[INFO] Coherence search over K = {list(K_RANGE)} "
          f"({K_SEARCH_PASSES} passes each)...")
    scores_df = search_k(corpus_bow, dictionary, token_lists)

    # Save scores
    scores_path = OUTPUT_DIR / "coherence_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"\n[INFO] Coherence scores saved → {scores_path.name}")
    print(scores_df.to_string(index=False))

    best_cv_k = int(scores_df.loc[scores_df["coherence_cv"].idxmax(), "K"])
    print(f"\n[INFO] Suggested K by C_v  : {best_cv_k}")
    print(f"[INFO] Suggested K by UMass: "
          f"{int(scores_df.loc[scores_df['coherence_umass'].idxmax(), 'K'])}")

    # Save plot
    save_coherence_plot(scores_df, OUTPUT_DIR / "coherence_plot.png")

    print("\n[INFO] Step 2 complete.")
    print()
    print("=" * 70)
    print("  !! HUMAN INTERVENTION REQUIRED — pipeline cannot continue automatically")
    print("=" * 70)
    print()
    print("  The final K cannot be chosen algorithmically. You must:")
    print()
    print("  1. Download from HPC:")
    print("       coherence_scores.csv")
    print("       coherence_plot.png")
    print()
    print("  2. Open notebooks/06_lda_inspection.ipynb locally.")
    print("     Inspect coherence metrics (u_mass, c_npmi) and diversity (PUW).")
    print("     Look at topic words per K — are they interpretable and distinct?")
    print()
    print(f"  3. Choose K.  Metric hints:")
    print(f"       Best C_v    → K = {best_cv_k}")
    print(f"       Best UMass  → K = {int(scores_df.loc[scores_df['coherence_umass'].idxmax(), 'K'])}")
    print(f"       Best NPMI   → K = {int(scores_df.loc[scores_df['coherence_npmi'].idxmax(), 'K'])}")
    print(f"     NOTE: metrics alone are not sufficient — interpretability is the")
    print(f"     primary criterion. Choose the K whose topics make financial sense.")
    print()
    print("  4. Open src/topic_modeling/config_lda.py and set:")
    print("       K_FINAL = <your_choice>")
    print()
    print("  5. ONLY THEN run Step 3:")
    print("       sbatch run_train_lda.sh")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
