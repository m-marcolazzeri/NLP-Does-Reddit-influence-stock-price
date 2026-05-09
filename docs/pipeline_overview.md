# Pipeline overview

How does a Reddit `.zst` archive become a comment-level dataset annotated for downstream financial-NLP analysis? Each section names the script, the input, the output, and the design choice behind the step.

## Stage 1 — Extraction

**Script:** `src/extraction/extract_reddit_matches.py` (uses `build_regex_patterns.py`).

The Pushshift archive is multi-GB and JSON-line, Zstandard-compressed. We stream it line by line, keeping only objects (comments or submissions) whose text mentions any of the tracked stocks.

Matching uses a tiered dictionary loaded from `config/alias_dictionary_v1.json`:
- cashtags (`$NVDA`)
- exact tickers (`NVDA`)
- company names (`Nvidia`)
- safe aliases / risky aliases (off by default)

For tickers that are easily confused with common English words (e.g. `META`, `AAPL`-as-`Apple`), an extra finance-context regex must also match the surrounding text. Match metadata (`match_sources`, `match_count`, `match_confidence`, `is_multi_match`, `needs_context_filter`) is preserved on every row so that downstream filtering can be revisited.

Output: `data/extraction/wsb_comments_2025.csv`.

## Stage 2 — Structural cleaning

**Script:** `src/extraction/clean_comments_structural.py`.

This stage is intentionally conservative. We do **not** filter by topic relevance here — semantic filtering is deferred until after submission context is attached and ultimately delegated to later stages. We only remove rows that are structurally broken or trivially noisy:

1. rows with missing or invalid `submission_id`
2. duplicate comment ids
3. authors that look like bots or moderators
4. composite duplicates `(submission_id, created_utc, body_text)`

We sort within each thread by `(submission_id, created_utc, id)` and add `thread_position` (1-based index inside the thread).

Output: `data/extraction/wsb_comments_2025_clean_structural.csv`, plus the helper file `required_submission_ids_from_comments_2025.csv` that lists the unique parent submission ids the cleaned comments depend on.

## Stage 3 — Recover parent submissions

**Script:** `src/extraction/recover_parent_submissions.py`.

The initial extraction only kept submissions that themselves matched the dictionary. But many comments mention a tracked stock while their parent submission does not (the comment was a side remark in an unrelated thread). To attach context to those comments, we stream the raw submissions `.zst` a second time and keep every submission whose id appears in the helper file from Stage 2.

Output: `data/extraction/wsb_submissions_2025_recovered.csv`. Match-related columns are `null` for submissions recovered this way (they did not pass through the matcher).

## Stage 4 — Merge

**Script:** `src/extraction/build_merged_dataset.py`.

We left-join the cleaned comments to the recovered submissions on `submission_id`, build a `submission_text` field by concatenating `title + selftext` (or just `title` if the body is empty/`[removed]`/`[deleted]`), and emit the comment-level analytical dataset.

This is the **main analytical artifact**: one row per comment, enriched with the parent thread's text. Submissions are not separate rows; they ride along as the `submission_text` column. The merge is sorted by `(submission_id, created_utc, thread_position)` so threads are contiguous.

Output: `data/extraction/wsb_merged_comments_with_submission.csv`.

## Stage 5 — Prepare LLM chunks

**Script:** `src/summarization/prepare_threads.py`.

Threads can have hundreds of comments, too long to feed to the model in one prompt. We group comments by `submission_id`, order them by `thread_position`, and split into fixed-size chunks of max 100 messages. Each chunk is formatted with explicit `[MSG 1]`, `[MSG 2]`, … markers so the model can refer to specific messages if needed.

Each output row is **one chunk of one thread**. The same `submission_text` is repeated on every chunk of the same thread for prompting convenience.

Output: `data/summarization/thread_chunks_v1.csv`.

## Stage 6 — Summarize threads

**Script:** `src/summarization/run_thread_summarizer.py` (prompts in `src/summarization/prompts.py`).

For each thread we iterate over its chunks in order:

- chunk 0 → `build_initial_thread_summary_prompt(submission_text, chunk_comments)` produces an initial structured JSON summary
- chunk N (N > 0) → `build_update_thread_summary_prompt(submission_text, previous_summary_json, chunk_comments)` produces an updated summary

Only the final summary (after the last chunk) is stored. The summary is a fixed-schema JSON with five fields: `main_stock_or_company`, `thread_topic`, `financial_angle`, `conversation_character`, `summary_for_labeling`. Allowed values for `financial_angle` and `conversation_character` are normalized post-hoc to guard against schema drift in model outputs. JSON parsing is robust: we extract the **last** valid `{...}` object from the model's text output to tolerate stray prose.

Default model: `Qwen/Qwen2.5-14B-Instruct`, deterministic decoding (`do_sample=False`). Checkpoints are written every 20 threads.

Output: `data/summarization/thread_summaries_all_chunks_v1.csv`. One row per thread.

**Why a summarizer rather than direct labeling?** Reddit comments are often telegraphic and only make sense in context. Labeling each comment in isolation systematically under-labels relevance. Producing a thread-level structured summary first lets the labeler resolve references like "this is bullish" by looking at the surrounding discussion, without re-feeding every thread in full to the labeler. The summarizer is a context-enrichment layer, **not** the final classifier.

## Stage 7 — Relevance filtering

**Script:** `src/relevance/predict_relevance_zeroshot.py`. Evaluation: `src/relevance/evaluate_bart.py`, `notebooks/04_bart_evaluation.ipynb`.

Each comment is assigned a label in `{1 = relevant, 0 = borderline, -1 = not relevant}`.

The classifier is `facebook/bart-large-mnli`, a zero-shot NLI model. Relevance is operationalized as a set of **hand-crafted hypothesis strings** (e.g. "This comment discusses the financial performance or stock price of a company"). The model scores each comment against the hypotheses and assigns a label based on asymmetric probability thresholds (0.75 for `relevant`, 0.50 for `not relevant`).

**Iterative calibration loop.** Classifier quality is validated against `data/relevance/rows_labeled_by_hand_3000.xlsx`, a set of 3,000 comments labeled by hand. After each run, `evaluate_bart.py` computes precision, recall, and F1 per class. If the metrics are unsatisfactory, the hypothesis strings are revised and the classifier is re-run; no model retraining is needed since the classifier is zero-shot.

**Evaluation results** (`data/relevance/bart_eval_report.json`, eval set = 3,000 hand-labeled rows):

| View | Precision (macro) | Recall (macro) | F1 (macro) | Cohen's κ |
|---|---|---|---|---|
| Strict (borderline → not relevant) | 0.72 | 0.65 | 0.64 | 0.32 |
| Liberal (borderline → relevant) | 0.85 | 0.71 | 0.73 | 0.48 |

Under the strict view, the classifier has high precision on "not relevant" (0.77) but low recall (0.37), meaning it misses many non-relevant comments. Under the liberal view, relevant recall reaches 0.98 at the cost of lower not-relevant recall (0.43). Per-stock macro-F1 is consistent: AMD 0.63, NVDA 0.63, PLTR 0.66. The pipeline uses `predicted_relevance == 1` as the downstream filter, which aligns with the liberal framing.

Output: `data/relevance/relevance_predictions_v1.csv` (239k rows). Only rows with `predicted_relevance == 1` proceed downstream.

## Stage 7b — Text cleaning

**Script:** `src/corpus_building/clean_text.py`.

Relevant comments (label `+1`) undergo a second, deeper cleaning pass before entering the NLP pipeline. The steps applied in order are:

1. Filter to `predicted_relevance == 1`
2. Remove Reddit-proprietary emotes (`![img](emote|...)`)
3. Strip spoiler markup, preserving inner text (`>!text!<` → `text`)
4. Strip Markdown links, preserving anchor text (`[text](url)` → `text`)
5. Remove residual bare URLs
6. Strip Markdown bold/italic markers, preserving text
7. Remove Reddit quote lines (lines beginning with `>`)
8. Remove user/subreddit mentions (`u/`, `r/`)
9. Remove inline bot commands (`!\w+`, e.g. `!banbet`, `!remindme`) — not caught by the NLI pre-filter when appearing mid-text
10. Convert emoji to words via `emoji.demojize` (e.g. 🚀 → "rocket") — appropriate for classical LDA, which has no embedding representation of symbols
11. Lowercase
12. Remove all digits — prices and figures are too document-specific to contribute to generalizable latent topics; semantic signal comes from words (bullish, dump, puts), not numbers
13. Retain only letters, spaces, `$`, `!`, `?`
14. Normalize whitespace
15. Deduplicate on `id`
16. Drop comments shorter than 5 words (`MIN_WORDS = 5`)

Output: `data/corpus_building/wsb_comments_clean_v1.csv`.

## Stage 8 — Corpus building

**Script:** `src/corpus_building/build_chunks.py`.

Individual Reddit comments are too short (typically 1–3 sentences) to provide a reliable statistical signal for topic modeling. Comments are therefore aggregated into pseudo-documents called **chunks** before LDA.

Two distinct outputs are produced, each with its own chunking logic, because they serve different purposes downstream.

**Training corpus** (`chunks_lda_v1.csv`): includes all relevant comments mentioning at least one target stock (including multi-stock mentions). Chunking is thread-based: threads with at least `MIN_SIZE` messages are chunked internally; shorter threads are pooled and chunked chronologically. This preserves the semantic coherence of discussions. Result: **6,635 chunks**, ~30 messages each.

**Inference corpus** (`chunks_sentiment_v1.csv`): includes only single-stock comments (where `matched_tickers` is exactly one of NVDA, AMD, or PLTR). Chunking is per-stock per-day, ordered chronologically. This is the corpus used to infer daily θ distributions for the financial panel. Result: **5,971 chunks**, 0 cross-date chunks.

| Parameter | Value | Rationale |
|---|---|---|
| `TARGET_SIZE` | 30 | Balances semantic richness against temporal granularity |
| `MIN_SIZE` | 15 | Minimum thread size before pooling short threads |

The thread-based vs. chronological split is intentional: training benefits from semantic coherence within discussions; inference requires temporal alignment with market returns.

## Stage 9 — LDA topic modeling

**Scripts:** `src/topic_modeling/01_build_corpus.py`, `02_search_k.py`, `03_train_lda.py`. Shared config: `src/topic_modeling/config_lda.py`. HPC scripts: `hpc/run_corpus.sh`, `hpc/run_search_k.sh`, `hpc/run_train_lda.sh`.

The pipeline has three steps following a human-in-the-loop design.

**Corpus construction** (`01_build_corpus.py`):

- Tokenization and lemmatization with spaCy (`en_core_web_sm`), retaining NOUN, VERB, ADJ, ADV tokens only
- Removal of spaCy stopwords, tokens shorter than 3 characters, ticker and company name aliases (from `config/alias_dictionary_v1.json`) — these appear by construction in every document and carry no discriminative topic signal
- Domain-specific stopword list (`DOMAIN_STOPWORDS`), built in two rounds: first pass removed generic terms (`chip`, `high`, `make`, `let`, `want`); second pass removed financially meaningful but non-discriminative terms (`trade`, `option`, `invest`, `gain`, `work`, `lot`, `well`, etc.) that appeared in 4–6 of 9 topics in earlier runs
- Bigram and trigram detection via Gensim Phrases (bigram `min_count=30`, `threshold=20`; trigram `min_count=20`, `threshold=15`) — conservative thresholds suppress noise bigrams while capturing domain phrases like `price_target` or `earnings_call`
- Dictionary filtering: `no_below=5` (token must appear in ≥5 chunks), `no_above=0.35` (token must appear in ≤35% of chunks; lowered from 0.50 after words like "price" at 44% appeared in 6–8 topics)
- Corpus serialized as raw **Bag-of-Words** (`corpus_bow.mm`); a TF-IDF version is saved for ablation only

**Why BoW and not TF-IDF:** LDA is a Dirichlet-multinomial generative model that requires raw integer word counts. Passing continuous TF-IDF weights distorts the posterior, causing *theta domination* — one topic absorbs >90% of the mass for most documents (empirically observed: mean max-θ ≈ 0.92 with TF-IDF vs. 0.54–0.71 with BoW).

**K search** (`02_search_k.py`): trains one LDA model per K in `{5, ..., 25}`, computes UMass, C_v, NPMI coherence, and PUW topic diversity. The pipeline stops here; a human inspects `coherence_scores.csv` and `coherence_plot.png`, chooses K, and sets `K_FINAL` in `config_lda.py` before proceeding. K=7 was selected: best UMass (−1.54) and NPMI (−0.032); C_v was flat across all K values; PUW diversity collapsed sharply at K≥9, signalling topic redundancy.

**Final training and inference** (`03_train_lda.py`): trains the final model with `K_FINAL=7` (15 passes, asymmetric α, symmetric η — `auto` is unsupported by `LdaMulticore` with multiple workers), saves topic word lists, and infers θ distributions on the sentiment chunks.

Output: `data/topic_modeling/topic_words_v1.csv`, `data/topic_modeling/results_v1.csv`, `data/topic_modeling/coherence_scores.csv`, `data/topic_modeling/coherence_plot.png`.

## Stage 10 — Financial panel and predictive modeling

**Scripts:** `src/modeling/` (not yet implemented).

Aggregate daily θ distributions per stock (mean across chunks for each stock-day), merge with market return data, and run short-term predictive models. Blocked on: completion of topic labeling and final LDA run.

## Design choices

- **Comment-level analytical unit.** Submissions are context, not data points. This avoids double-counting and matches the granularity of downstream sentiment/topic work.
- **Conservative early cleaning.** Stage 2 does only what is structurally justified. Topic-level filtering is deferred to the NLI stage where richer context is available.
- **Two-pass extraction for parent recovery.** Streaming the raw archive twice is preferable to keeping every submission in memory or dropping comments whose parent didn't match the dictionary.
- **Chunked sequential summarization.** Avoids prompt-length explosion for long threads while preserving narrative continuity, at the cost of one LLM call per chunk.
- **Thread-based chunking for training, chronological for inference.** Training benefits from the semantic coherence of discussions; inference requires temporal alignment with market returns.
- **BoW over TF-IDF for LDA.** Empirically validated: TF-IDF causes theta domination; BoW yields well-distributed topic proportions.
- **Versioned artifacts.** `_v1` filenames pin prompt, model, and parameter versions. Incrementing the suffix is the convention when any of these change.
