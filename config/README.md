# config/

Static configuration files shared across pipeline stages. Loaded at runtime by scripts in `src/`; not modified by any pipeline step.

| File | Purpose |
|---|---|
| `stock_universe_v1.json` | Defines the 10-stock extraction universe (AAPL, MSFT, NVDA, AMD, AMZN, GOOGL, META, TSLA, PLTR, INTC) and the 2025-01-01 → 2025-12-31 time window. The downstream NLP universe (NVDA, AMD, PLTR) is not defined here — it is applied by `filter_to_3stocks.py`. |
| `extraction_settings_v1.json` | Controls the regex matching strategy: which match sources are active (cashtags, exact tickers, company names), whether risky/safe aliases are used, and what metadata columns to save. |
| `alias_dictionary_v1.json` | Maps each ticker to its recognized aliases, common misspellings, and cashtag variants used to build the extraction regex patterns. |
| `extracted_row_schema_v1.json` | JSON Schema for a single extracted row; used to validate extraction output and document the column set. |
