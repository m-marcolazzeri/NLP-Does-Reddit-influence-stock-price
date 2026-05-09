# Reddit Tech Stocks NLP

Academic NLP project. We extract Reddit discussions about a fixed universe of US tech stocks from the r/wallstreetbets archive, build a comment-level dataset enriched with submission context, summarize each thread with an LLM, and (in later stages) use those summaries to label individual comments by financial relevance for downstream sentiment, topic modeling, aggregation, and short-term market-activity prediction.

## Scope

- Subreddit: r/wallstreetbets
- Time window (current pipeline): 2025-01-01 → 2025-12-31.
- Stock universe (extraction): AAPL, MSFT, NVDA, AMD, AMZN, GOOGL, META, TSLA, PLTR, INTC. (See `config/stock_universe_v1.json`.)
- Stock universe (NLP + modeling): focus on **NVDA, AMD, PLTR** only.

## Pipeline at a glance

| Stage | Purpose | Code | Input | Output |
|---|---|---|---|---|
| 1. Extract | Stream-decompress raw `.zst` archives, keep only rows that mention any tracked stock. | `src/extraction/extract_reddit_matches.py` (+ `build_regex_patterns.py`) | `data/raw/*.zst` | `data/extraction/wsb_comments_2025.csv` |
| 2. Structural cleaning | Drop rows with bad `submission_id`, deduplicate, drop bot/mod authors, sort threads, emit `thread_position`. | `src/extraction/clean_comments_structural.py` | `data/extraction/wsb_comments_2025.csv` | `data/extraction/wsb_comments_2025_clean_structural.csv` + `required_submission_ids_from_comments_2025.csv` + `data/extraction/reports/...json` |
| 3. Recover parent submissions | Stream the raw submissions file to recover parent submissions by id. | `src/extraction/recover_parent_submissions.py` | `data/raw/...submissions.zst` + IDs file from stage 2 | `data/extraction/wsb_submissions_2025_recovered.csv` |
| 4. Merge | Attach each submission's text to every one of its comments (one row = one comment). | `src/extraction/build_merged_dataset.py` | cleaned comments + recovered submissions | `data/extraction/wsb_merged_comments_with_submission.csv` |
| 5. Summarize threads | Chunk threads into 100-message windows, sequentially summarize with Qwen2.5-14B-Instruct, build relevance input. | `src/summarization/` | merged dataset | `data/summarization/thread_summaries_all_chunks_v1.csv`, `data/relevance/pre_relevance_v1.csv` |
| 6. Relevance labeling | BART zero-shot NLI classifier assigns `predicted_relevance ∈ {1, 0, -1}` to each comment. Evaluated against 3 000 hand-labeled rows. | `src/relevance/predict_relevance_zeroshot.py`, `src/relevance/evaluate_bart.py` | `data/relevance/pre_relevance_v1.csv` | `data/relevance/relevance_predictions_v1.csv` |
| 7. Corpus building | Text cleaning (spaCy lemmatization, stopwords) + aggregation into per-stock-per-day chunks for LDA. | `src/corpus_building/` | `data/relevance/relevance_predictions_v1.csv` | `data/corpus_building/chunks_lda_v1.csv`, `chunks_sentiment_v1.csv` |
| 8. LDA topic modeling | bigrams/trigrams → BoW corpus → coherence-based K search → final LDA training (K=7) + θ inference. | `src/topic_modeling/` (3 scripts + `config_lda.py`) | `data/corpus_building/chunks_lda_v1.csv`, `chunks_sentiment_v1.csv` | `data/topic_modeling/topic_words_v1.csv`, `results_v1.csv` |
| 9. Financial panel + modeling | (planned) Aggregate daily θ per stock, merge with market returns, run predictive models. | `src/modeling/` | `data/topic_modeling/results_v1.csv` + price data |  |

The **main analytical artifact** is the comment-level merged dataset (stage 4). Submissions are not separate rows; they are attached as the `submission_text` context column.

## Repository layout

```
config/                  Extraction settings, alias dictionary, stock universe, output schema
data/
  raw/                   Raw .zst archives (not tracked)
  extraction/            Matched comments, cleaned, recovered submissions, merged dataset
  summarization/         Thread chunks and LLM summaries
  relevance/             pre_relevance input, BART predictions, hand-labeled evaluation sets
  corpus_building/       Cleaned text and per-stock-per-day chunks for LDA
  topic_modeling/        BoW corpus, LDA models, coherence scores, topic words, θ results
  modeling/              (planned) Financial panel and model outputs
docs/
  extraction/            Extraction roadmap
  relevance/             Annotation guidelines, zero-shot NLI architecture
  topic_modeling/        LDA pipeline summary
  pipeline_overview.md   End-to-end narrative (all stages)
  session_handoff.md     Current status and open issues
notebooks/               Inspection notebooks, numbered in pipeline order (01–07)
src/
  extraction/            Stages 1–4: extract, clean, recover, merge
  summarization/         Stage 5: thread chunking, LLM summarization, pre_relevance build
  relevance/             Stage 6: BART classifier, evaluation
  corpus_building/       Stage 7: text cleaning + chunking
  topic_modeling/        Stage 8: LDA pipeline (config_lda.py + 3 scripts)
  modeling/              Stage 9: financial panel (planned)
```

## Running end-to-end

```bash
# Stages 1–4: extraction pipeline
python src/extraction/extract_reddit_matches.py \
  --input data/raw/subreddits25/wallstreetbets_comments.zst \
  --output data/extraction/wsb_comments_2025.csv \
  --source-type comment --start-date 2025-01-01 --end-date 2025-12-31

python src/extraction/clean_comments_structural.py
python src/extraction/recover_parent_submissions.py
python src/extraction/build_merged_dataset.py

# Stage 5: summarization (HPC recommended)
python src/summarization/prepare_threads.py
python src/summarization/run_thread_summarizer.py
python src/summarization/build_relevance_input.py

# Stage 6: relevance (HPC recommended)
python src/relevance/predict_relevance_zeroshot.py

# Stage 7: corpus building
python src/corpus_building/clean_text.py
python src/corpus_building/build_chunks.py

# Stage 8: LDA (HPC recommended)
sbatch hpc/run_corpus.sh      # 01_build_corpus.py
sbatch hpc/run_search_k.sh    # 02_search_k.py  → inspect notebooks/06_lda_inspection.ipynb
sbatch hpc/run_train_lda.sh   # 03_train_lda.py
```

## Setup

```bash
pip install -r requirements.txt
```

Stage 5–6 need a CUDA-capable GPU and ~30 GB of VRAM for the default model (Qwen2.5-14B-Instruct).

## Documentation

- `docs/pipeline_overview.md` — narrative description of every stage and design choice.
