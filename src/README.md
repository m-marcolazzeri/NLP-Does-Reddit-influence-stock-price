# src/

Source code for the Reddit Tech Stocks NLP pipeline. Each subfolder is one pipeline stage and an importable Python package (`__init__.py` present). Deprecated or superseded scripts are kept in `_archive/` subdirectories for reference.

## Stage map

| Folder | Stages | Purpose |
|---|---|---|
| `extraction/` | 1–4 | Stream `.zst` archives, filter to target stocks, structural cleaning, submission recovery, merge |
| `summarization/` | 5 | Chunk threads into 100-message windows, run LLM summarizer (Qwen2.5-14B-Instruct), build relevance input |
| `relevance/` | 6 | Zero-shot NLI relevance classifier (BART-large-MNLI) + evaluation against hand labels |
| `corpus_building/` | 7 | Text cleaning (spaCy lemmatization, stopwords) + per-stock-per-day chunking for LDA |
| `topic_modeling/` | 8 | Bigrams/trigrams → BoW corpus → coherence K search → LDA training + θ inference |
| `modeling/` | 9 | Financial panel + predictive models (not yet implemented) |

Scripts are run from the project root (`python src/<stage>/<script>.py`). Paths in each script are resolved relative to the project root via `Path(__file__).resolve().parents[2]`.

---

## extraction/

| Script | Role |
|---|---|
| `build_regex_patterns.py` | Compiles per-ticker regex patterns from the alias dictionary and extraction settings |
| `extract_reddit_matches.py` | Streams the raw `.zst` comment archive; keeps rows mentioning any of the 10 extraction-universe tickers |
| `filter_to_3stocks.py` | Filters the 10-stock extract (~519k rows) to NVDA, AMD, PLTR only (~248k rows); input to stage 2 |
| `clean_comments_structural.py` | Drops bad/duplicate/bot rows, sorts threads, emits `thread_position` and a JSON QC report |
| `recover_parent_submissions.py` | Streams the raw submissions `.zst` to recover parent posts by `submission_id` |
| `build_merged_dataset.py` | Joins each submission's text onto its comments; output is the main analytical artifact |
| `_archive/` | `make_clean_single_ticker_dataset.py`, `score_finance_relevance.py` (early prototypes) |

## summarization/

| Script | Role |
|---|---|
| `prepare_threads.py` | Groups comments by thread, splits into sequential 100-message chunks, formats with `[MSG N]` markers |
| `run_thread_summarizer.py` | Runs Qwen2.5-14B-Instruct on each chunk; each chunk receives the previous chunk's summary as context |
| `build_remaining_chunks.py` | Utility to resume a partial summarization run without reprocessing completed chunks |
| `build_relevance_input.py` | Attaches the LLM summary as `LLM_summary` context to each comment; produces the classifier input CSV |
| `prompts.py` | Prompt templates (chunk-0 and chunk-N variants) |

## relevance/

| Script | Role |
|---|---|
| `predict_relevance_zeroshot.py` | BART zero-shot NLI classifier; assigns `predicted_relevance ∈ {1, 0, -1}` to each comment using asymmetric confidence thresholds (0.75 relevant / 0.50 not-relevant) |
| `evaluate_bart.py` | Computes classification report and confusion matrix vs `rows_labeled_by_hand_3000.xlsx` |
| `rebuild_pre_relevance.py` | Regenerates `pre_relevance_v1.csv` if the merged dataset is updated |
| `_archive/` | `predict_relevance.py` (supervised prototype), `train_relevance_classifier.py` |

## corpus_building/

| Script | Role |
|---|---|
| `clean_text.py` | Filters to `predicted_relevance == 1` (~195k rows), lowercases, strips URLs/punctuation, lemmatizes with spaCy, applies domain stopwords |
| `build_chunks.py` | Aggregates cleaned tokens into per-stock-per-day documents; emits separate LDA and sentiment chunk files |

## topic_modeling/

| Script | Role |
|---|---|
| `config_lda.py` | All LDA hyperparameters and I/O paths — single source of truth imported by the three pipeline scripts |
| `01_build_corpus.py` | Trains Gensim Phrases (bigrams/trigrams), builds Dictionary and BoW corpus. Also builds a TF-IDF corpus saved as `corpus_tfidf_reference.mm` for ablation comparison only — it is **not** fed to LDA. |
| `02_search_k.py` | Sweeps K; computes u_mass, c_v, c_npmi coherence and PUW topic diversity. **Ends with a mandatory human-in-the-loop step** — see below. |
| `03_train_lda.py` | Trains final LDA (`LdaMulticore`); infers per-document θ distributions |

> **Human-in-the-loop checkpoint between `02_search_k.py` and `03_train_lda.py`**
>
> After `02_search_k.py` finishes, the pipeline cannot continue automatically. A researcher must:
> 1. Inspect `coherence_scores.csv` and `coherence_plot.png`.
> 2. Open `notebooks/06_lda_inspection.ipynb` and inspect coherence + diversity metrics and topic word quality across candidate K values.
> 3. Choose K based on a combination of quantitative metrics and qualitative interpretability.
> 4. Set `K_FINAL = <chosen_K>` in `config_lda.py`.
> 5. Only then run `03_train_lda.py`.
>
> `02_search_k.py` prints an explicit alert at termination as a reminder.