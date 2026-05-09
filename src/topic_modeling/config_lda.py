"""
config_lda.py
─────────────
Single source of truth for all LDA pipeline parameters.
Imported by 01_build_corpus.py, 02_search_k.py, 03_train_lda.py.

Human-in-the-loop parameters
─────────────────────────────
After running 02_search_k.py and inspecting the coherence plots in
notebook 06_lda_inspection.ipynb, set K_FINAL to the chosen number of
topics, then re-run 03_train_lda.py.

To iterate on topic quality without re-building the corpus:
  1. Adjust K_FINAL, LDA_PASSES, ALPHA, ETA, or TOPIC_TOP_N below.
  2. Re-run: sbatch run_train_lda.sh   (fast — ~2 h)
  3. Inspect output in notebook 04.
  4. Repeat until topics are interpretable.
"""

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_LDA_CHUNKS       = BASE_DIR / "data/corpus_building/chunks_lda_v1.csv"
INPUT_SENTIMENT_CHUNKS = BASE_DIR / "data/corpus_building/chunks_sentiment_v1.csv"
OUTPUT_DIR             = BASE_DIR / "data/topic_modeling"
ALIAS_DICT_PATH        = BASE_DIR / "config/alias_dictionary_v1.json"

# ---------------------------------------------------------------------------
# Alias filter
# ---------------------------------------------------------------------------

# Stocks whose ticker/company aliases are excluded from the LDA vocabulary.
# These words appear by construction in every document (we filtered to these
# stocks), so they carry no discriminative topic information.
ALIAS_TARGET_STOCKS = {"NVDA", "AMD", "PLTR"}

# Fields of alias_dictionary_v1.json to use for filtering.
# "notes" and "context_keywords" are intentionally excluded.
ALIAS_FIELDS = ("exact_tickers", "company_names", "safe_aliases", "risky_aliases")

# ---------------------------------------------------------------------------
# Step 1 — Tokenisation & vocabulary
# ---------------------------------------------------------------------------

ALLOWED_POS   = {"NOUN", "VERB", "ADJ", "ADV"}   # POS tags kept after spaCy
MIN_TOKEN_LEN = 3                                  # drop tokens shorter than N chars

# Domain-specific stopwords on top of spaCy's built-in list.
# Words that appear in 20–35 % of chunks (below no_above) but carry no
# topic-discriminative signal in a Reddit semiconductor corpus.
#
# Round 1 additions (corpus-ubiquitous after DICT_NO_ABOVE=0.35 fix):
#   chip   – appears in 6/9 topics; generic semiconductor term present by
#             construction in an NVDA/AMD corpus
#   high   – ambiguous filler ("price is high", "all-time high")
#   make   – generic verb with no topical content
#   let    – generic Reddit opener ("let me...", "let's...")
#   want   – generic desire verb ("I want to...", "do you want...")
#
# Round 2 additions (post-run inspection — filler words spanning 4-6/9 topics):
#   trade  – appears in 6/9 topics; too generic to distinguish topics
#   work   – appears in 6/9 topics; generic verb ("it works", "working on")
#   lot    – appears in 5/9 topics; quantifier filler ("a lot of...")
#   well   – appears in 4/9 topics; discourse filler ("as well", "well...")
#   option – appears in 5/9 topics; financial term but overloaded
#             (equity options + Reddit "any option?") — collapses 4 topics
#   invest – appears in 5/9 topics; too broad across all financial topics
#   gain   – appears in 5/9 topics; generic outcome word
#   point  – appears in 4/9 topics; discourse filler ("my point is...")
#   low    – appears in 4/9 topics; ambiguous ("buy low", "all-time low")
#   win    – appears in 3/9 topics; vague outcome filler
#   bet    – appears in 3/9 topics; Reddit discourse ("I bet", "my bet")
#   actually – discourse filler (ADV — passes POS filter but no signal)
#   probably – discourse filler (ADV — passes POS filter but no signal)
DOMAIN_STOPWORDS: set = {
    # round 1
    "chip", "high", "make", "let", "want",
    # round 2
    "trade", "work", "lot", "well", "option",
    "invest", "gain", "point", "low",
    "win", "bet", "actually", "probably",
}

# ---------------------------------------------------------------------------
# Step 1 — Bigram / Trigram detection (Gensim Phrases)
# ---------------------------------------------------------------------------

BIGRAM_MIN_COUNT  = 30   # pair must co-occur >= N times to form a bigram
                         # (raised from 10 to suppress emoji-derived bigrams)
BIGRAM_THRESHOLD  = 20   # higher = stricter (fewer bigrams formed)

# Trigrams: second Phrases pass on already-bigrammed tokens.
# Lower min_count is appropriate because trigram candidates are rarer.
TRIGRAM_MIN_COUNT = 20
TRIGRAM_THRESHOLD = 15

# ---------------------------------------------------------------------------
# Step 1 — Gensim Dictionary filtering
# ---------------------------------------------------------------------------

DICT_NO_BELOW = 5     # token must appear in >= N chunks (~0.08 % of 6 500 docs)
DICT_NO_ABOVE = 0.35  # token must appear in <= 35 % of chunks
                      # Lowered from 0.50: words like "price" (44 %),
                      # "people" (50 %), "way" (45 %), "tomorrow" (43 %),
                      # "thing" (33 %), "big" (42 %) survived the old filter
                      # but appeared in 6–8 / 9 topics → non-discriminative.

# ---------------------------------------------------------------------------
# Step 2 — Coherence search (K optimisation)
# ---------------------------------------------------------------------------

K_RANGE        = range(5, 26, 2)   # K values tested: 5, 7, 9, …, 25
K_SEARCH_PASSES = 5                # fewer passes for K search (speed vs. accuracy)
                                   # final model uses LDA_PASSES (more passes)

# ---------------------------------------------------------------------------
# Step 3 — Final LDA training
# ⬇  HUMAN SETS THIS after inspecting 02_search_k output  ⬇
# ---------------------------------------------------------------------------

K_FINAL: Optional[int] = 7        # ← set after inspecting coherence plots
                                   # if None: auto-select best c_v from search
                                   # Changed 9→7: Topics 1,3,5,7 were near-identical
                                   # (all options-trading vocabulary). Lowering K
                                   # collapses redundant topics into one.

LDA_PASSES = 15      # passes over corpus during final training
WORKERS    = 3       # LdaMulticore workers (= n_cpu_cores − 1 on the HPC node)
ALPHA      = "asymmetric"   # Dirichlet α prior — 'asymmetric' = 1/K per topic
                             # NOTE: 'auto' is NOT supported by LdaMulticore
ETA        = "symmetric"    # Dirichlet η prior — 'symmetric' = 1/|V|
                             # NOTE: 'auto' is NOT supported by LdaMulticore
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Step 3 — Output
# ---------------------------------------------------------------------------

TOPIC_TOP_N       = 20   # words per topic saved to topic_words_v1.csv
                          # can be changed WITHOUT retraining (just re-run step 3)
TOPIC_DIVERSITY_N = 10   # top-N words used for diversity (PUW) calculation
