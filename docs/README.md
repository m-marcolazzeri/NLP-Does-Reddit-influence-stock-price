# `docs/`

Methodological and structural documentation for the project. Organized by pipeline stage.

| File / Folder | What it covers |
|---|---|
| `pipeline_overview.md` | End-to-end narrative of every stage in the pipeline, the artifacts each stage produces, and the design decisions behind them. |
| `session_handoff.md` | Handoff note for continuing work in a new session: logic behind key decisions, current status, and prioritized list of open issues. |
| `extraction/extraction_roadmap.md` | Methodology note describing the four-step extraction approach (pilot → audit → baseline → scale-up) used to build the production dictionary. |
| `relevance/relevance_scoring_usage.md` | How to invoke the heuristic relevance scorer (`src/extraction/_archive/score_finance_relevance.py`). |
| `relevance/zero_shot_nli_architecture.md` | Architecture and design notes for the BART zero-shot NLI classifier. |
| `relevance/LLM_relevance_summary.md` | LLM-assisted relevance scoring summary and methodology notes. |
| `relevance/relevance_classifier_readme.md` | Overview of the relevance classification system and label schema. |
| `relevance/llm_relevance_readme.md` | Design notes for using LLM summaries as classifier context. |
| `topic_modeling/pipeline_summary.md` | Detailed Italian-language summary of all LDA pipeline phases including parameters, architectural decisions, and open issues. The most up-to-date operational reference for Stage 8. |
