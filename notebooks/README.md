# notebooks/

Inspection and analysis notebooks, numbered in pipeline order. All notebooks assume the project root is on `sys.path` and that the corresponding data files are present.

## Canonical notebooks

| Notebook | Pipeline stage | Purpose |
|---|---|---|
| `01_data_inspection.ipynb` | Stage 1–2 | Inspect the raw extraction output (`wsb_comments_2025.csv`): ticker distribution, date coverage, row counts |
| `02_final_merged_dataset_inspection.ipynb` | Stage 4 | Validate the merged comment+submission dataset: schema, null rates, submission text coverage |
| `03_pipeline_3stocks_inspection.ipynb` | Stage 1b | Verify the 3-stock filter: coverage of NVDA, AMD, PLTR before and after `filter_to_3stocks.py` |
| `04_bart_evaluation.ipynb` | Stage 6 | Interactive review of BART relevance predictions vs hand labels; confusion matrix, error analysis |
| `05_vocabulary_inspection.ipynb` | Stage 7–8 | Examine cleaned token distributions, bigram/trigram candidates, domain stopword decisions |
| `06_lda_inspection.ipynb` | Stage 8 | Coherence scores by K, topic–word distributions, θ distribution diagnostics |
| `07_topic_labeling.ipynb` | Stage 8 | Assign human-readable labels to the 7 LDA topics; export to `topic_words_v1.csv` |
