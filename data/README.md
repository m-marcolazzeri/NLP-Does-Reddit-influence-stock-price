# data/

All datasets produced or consumed by the pipeline. Subfolders mirror the stage structure in `src/`. The primary analytical unit throughout is the **comment**: one row = one comment, `id` (Reddit comment ID) is the primary key, `submission_id` is the foreign key to the parent thread. All timestamps are UTC (`created_utc` epoch integer, `date_utc` YYYY-MM-DD string). List-type columns (`matched_tickers`, `matched_terms`) use `|` as an internal separator.

## Data lineage

```
data/raw/*.zst                                    [not tracked by git]
   │ extract_reddit_matches.py
   ▼
data/extraction/wsb_comments_2025.csv             10 stocks, ~519k rows
   │ filter_to_3stocks.py
   ▼
data/extraction/wsb_comments_3stocks.csv          NVDA/AMD/PLTR only, ~248k rows
   │ clean_comments_structural.py
   ▼
data/extraction/wsb_comments_2025_clean_structural.csv
data/extraction/required_submission_ids_from_comments_2025.csv
data/extraction/reports/wsb_comments_2025_clean_structural_report.json
   │ recover_parent_submissions.py
   ▼
data/extraction/wsb_submissions_2025_recovered.csv
   │ build_merged_dataset.py
   ▼
data/extraction/wsb_merged_comments_with_submission.csv   ← main analytical artifact
   │ prepare_threads.py
   ▼
data/summarization/thread_chunks_v1.csv
   │ run_thread_summarizer.py
   ▼
data/summarization/thread_summaries_all_chunks_v1.csv
   │ build_relevance_input.py
   ▼
data/relevance/pre_relevance_v1.csv
   │ predict_relevance_zeroshot.py
   ▼
data/relevance/relevance_predictions_v1.csv
   │ clean_text.py
   ▼
data/corpus_building/cleaned_v1.csv               predicted_relevance == 1 only, ~195k rows
   │ build_chunks.py
   ▼
data/corpus_building/chunks_lda_v1.csv            per-stock-per-day, LDA tokens
data/corpus_building/chunks_sentiment_v1.csv      per-stock-per-day, sentiment tokens
   │ 01_build_corpus.py
   ▼
data/topic_modeling/dictionary.gensim
data/topic_modeling/corpus_bow.mm
   │ 02_search_k.py
   ▼
data/topic_modeling/coherence_scores.csv
data/topic_modeling/coherence_plot.png
   │ 03_train_lda.py  (K=7)
   ▼
data/topic_modeling/model/lda_k7/
data/topic_modeling/topic_words_v1.csv
data/topic_modeling/results_v1.csv                per-document θ distributions
```

---

## raw/

Original Pushshift Reddit archives (`.zst` compressed NDJSON). Not tracked by git (see `.gitignore`). See `data/raw/README.md` for expected filenames and download instructions.

## extraction/

Outputs of stages 1–4.

| File | Description |
|---|---|
| `wsb_comments_2025.csv` | Raw extraction output: all comments from r/wallstreetbets 2025 mentioning any of the 10 tracked tickers |
| `wsb_comments_3stocks.csv` | Filtered to NVDA, AMD, PLTR; input to structural cleaning |
| `wsb_comments_2025_clean_structural.csv` | After deduplication, bot removal, thread sorting; includes `thread_position` |
| `required_submission_ids_from_comments_2025.csv` | Unique `submission_id` values needed to recover parent posts |
| `wsb_submissions_2025_recovered.csv` | Parent submission metadata + text for every thread in the cleaned comment set |
| `wsb_merged_comments_with_submission.csv` | **Main analytical artifact.** One row per comment with `submission_text` attached as context column |
| `reports/` | JSON quality-control reports from `clean_comments_structural.py` (row counts, drop reasons) |

## summarization/

Outputs of stage 5.

| File | Description |
|---|---|
| `thread_chunks_v1.csv` | Comments split into 100-message sequential chunks; columns: `submission_id`, `chunk_id`, `formatted_text` |
| `thread_summaries_all_chunks_v1.csv` | One row per `(submission_id, chunk_id)` with the LLM-generated summary; join key to the comment dataset: `chunk_id = (thread_position - 1) // 100` |

## relevance/

Inputs, outputs and evaluation data for stage 6.

| File | Description |
|---|---|
| `pre_relevance_v1.csv` | Classifier input: merged comments with `LLM_summary` attached as context |
| `relevance_predictions_v1.csv` | Full dataset with `predicted_relevance ∈ {1, 0, -1}` and `relevance_confidence` columns |
| `rows_labeled_by_hand_3000.xlsx` | 3 000 manually annotated comments used to evaluate the classifier; ground-truth label column: `Relevance` |
| `rows_labeled_by_hand_200.xlsx` | Earlier smaller annotation set (superseded) |
| `relevance_eval_results.csv` | BART predictions on the 3 000-row eval set (output of `evaluate_bart.py`) |
| Other `.xlsx` files | Sampling and inspection artifacts from annotation development |

## corpus_building/

Outputs of stage 7.

| File | Description |
|---|---|
| `cleaned_v1.csv` | Lemmatized, stopword-filtered tokens; one row per comment; `predicted_relevance == 1` only |
| `chunks_lda_v1.csv` | Per-stock-per-day token lists for LDA (domain stopwords applied) |
| `chunks_sentiment_v1.csv` | Per-stock-per-day token lists for sentiment analysis (lighter stopword list) |

## topic_modeling/

Outputs of stage 8.

| File / Folder | Description |
|---|---|
| `dictionary.gensim` | Gensim `Dictionary` mapping token → integer ID |
| `corpus_bow.mm` | Market-Matrix serialized bag-of-words corpus — the actual LDA training input |
| `corpus_tfidf_reference.mm` | TF-IDF weighted corpus — **reference/ablation only, not used by LDA**. LDA requires raw integer counts; passing TF-IDF weights distorts the Dirichlet-multinomial posterior (theta domination). Kept to support the methodological comparison in the report. |
| `tfidf_reference_model.gensim` | Trained TF-IDF model — reference only |
| `bigram_model.pkl`, `trigram_model.pkl` | Trained Gensim `Phrases` models |
| `vocab_cache.pkl` | Intermediate vocabulary cache used during corpus construction |
| `coherence_scores.csv` | u_mass / c_v / c_npmi / PUW scores per K value |
| `coherence_plot.png` | Coherence vs K visualization |
| `model/lda_k7/` | **Final model.** Gensim `LdaMulticore` serialized files |
| `model/lda_k9/`, `model/lda_k11/`, `model/lda_k21/` | Alternative K runs kept for comparison |
| `topic_words_v1.csv` | Top words per topic for K=7 |
| `results_v1.csv` | Per-document θ distributions (topic proportions) for K=7; input to stage 9 |

## modeling/

Empty placeholder for stage 9 (financial panel + predictive models).
