"""
prompts.py

Centralized prompt templates for the LLM pipeline.

Functions
---------
build_initial_thread_summary_prompt     [v1 - deprecated, kept for compatibility]
build_update_thread_summary_prompt      [v1 - deprecated, kept for compatibility]

build_chunk_summary_prompt              [v2 - chunk 0, no previous context]
build_chunk_summary_with_context_prompt [v2 - chunk N>0, previous chunk summary as context]

v2 design: one summary per chunk (not per thread). The output of each chunk is
specific to those 100 messages. The previous chunk summary is passed as context
to help interpret references and conversational continuity, but the output
describes only what is happening in the current chunk.
"""


def build_initial_thread_summary_prompt(submission_text: str, chunk_comments: str) -> str:
    return f"""
You are assisting an academic NLP project on Reddit discussions about stocks.

Your task is to produce a structured thread summary.
You are NOT classifying individual comments.
You are NOT deciding whether comments are relevant or not relevant.

You must summarize the discussion using only the information contained in:
1. the submission text
2. the current chunk of comments

Return exactly ONE JSON object and nothing else.

Field definitions:
- "main_stock_or_company": list the main stocks/companies explicitly. If there is more than one important entity, return them separated by " | ". Examples: "NVDA", "TSLA | NVDA", "AMD | INTC | PLTR". Use "unclear" only if no meaningful stock/company/entity can be identified.
- "thread_topic": one short sentence describing what the discussion is mainly about.
- "financial_angle": choose EXACTLY ONE of the following values:
  ["price_action", "trading_positions", "earnings", "valuation", "company_news", "sector_macro", "product_or_company_discussion", "mixed", "unclear"]
- "conversation_character": choose EXACTLY ONE of the following values:
  ["analytical", "speculative", "reactive", "meme_heavy", "mixed", "off_topic"]
- "summary_for_labeling": a compact contextual summary of at most 200 words. It should contain only the minimum context needed to interpret individual comments later.

Rules:
- If the thread is about a platform, instrument, or specific company that is central to interpreting the discussion, return that entity rather than "unclear". Use "unclear" only when no central entity can reasonably be identified.
- Do not write a long explanation.
- Do not restate all details.
- Prefer compression over completeness.
- Do not repeat the prompt.
- Do not repeat the submission text or comment chunk verbatim.
- Do not output markdown.
- Do not output explanations before or after the JSON.
- If something is unclear, use "unclear".
- Keep the JSON valid and compact.

SUBMISSION TEXT:
{submission_text}

CURRENT COMMENT CHUNK:
{chunk_comments}

Return exactly this JSON structure:
{{"main_stock_or_company":"...","thread_topic":"...","financial_angle":"...","conversation_character":"...","brief_summary":"..."}}
""".strip()


def build_update_thread_summary_prompt(
    submission_text: str,
    previous_summary_json: str,
    chunk_comments: str
) -> str:
    return f"""
You are assisting an academic NLP project on Reddit discussions about stocks.

Your task is to UPDATE an existing structured thread summary.
You are NOT classifying individual comments.
You are NOT deciding whether comments are relevant or not relevant.

You are given:
1. the submission text
2. the previous structured summary of the thread so far
3. the next chunk of comments

Update the summary so that it reflects the thread more completely.

Return exactly ONE JSON object and nothing else.

Field definitions:
- "main_stock_or_company": list the main stocks/companies explicitly. If there is more than one important entity, return them separated by " | ". Examples: "NVDA", "TSLA | NVDA", "AMD | INTC | PLTR". Use "unclear" only if no meaningful stock/company/entity can be identified.
- "thread_topic": one short sentence describing what the discussion is mainly about.
- "financial_angle": choose EXACTLY ONE of the following values:
  ["price_action", "trading_positions", "earnings", "valuation", "company_news", "sector_macro", "product_or_company_discussion", "mixed", "unclear"]
- "conversation_character": choose EXACTLY ONE of the following values:
  ["analytical", "speculative", "reactive", "meme_heavy", "mixed", "off_topic"]
- "summary_for_labeling": a compact contextual summary of at most 200 words. It should contain only the minimum context needed to interpret individual comments later.

Rules:
- If the thread is about a platform, instrument, or specific company that is central to interpreting the discussion, return that entity rather than "unclear". Use "unclear" only when no central entity can reasonably be identified.
- Preserve useful earlier context when still valid.
- Update the summary if the thread focus shifts.
- Do not write a long explanation.
- Do not restate all details.
- Prefer compression over completeness.
- Do not repeat the prompt.
- Do not repeat the submission text or comment chunk verbatim.
- Do not output markdown.
- Do not output explanations before or after the JSON.
- If something is unclear, use "unclear".
- Keep the JSON valid and compact.

SUBMISSION TEXT:
{submission_text}

PREVIOUS THREAD SUMMARY:
{previous_summary_json}

NEW COMMENT CHUNK:
{chunk_comments}

Return exactly this JSON structure:
{{"main_stock_or_company":"...","thread_topic":"...","financial_angle":"...","conversation_character":"...","brief_summary":"..."}}
""".strip()


# ---------------------------------------------------------------------------
# v2 prompts — one summary per chunk (not per thread)
# ---------------------------------------------------------------------------

def build_chunk_summary_prompt(submission_text: str, chunk_comments: str) -> str:
    """
    Prompt for chunk 0 (no previous context available).

    Generates a summary specific to this chunk of 100 messages.
    """
    return f"""
You are assisting an academic NLP project on Reddit discussions about stocks.

Your task is to summarize a specific group of comments from a Reddit thread.
You are NOT classifying individual comments.
You are NOT deciding whether comments are relevant or not relevant.

Return exactly ONE JSON object and nothing else.

Field definitions:
- "main_stock_or_company": stocks/companies explicitly discussed in this group, pipe-separated. Use "unclear" only if none can be identified.
- "thread_topic": one sentence on what the submission is about.
- "financial_angle": dominant financial angle of THIS group. Choose EXACTLY ONE:
  ["price_action", "trading_positions", "earnings", "valuation", "company_news", "sector_macro", "product_or_company_discussion", "mixed", "unclear"]
- "conversation_character": dominant tone of THIS group. Choose EXACTLY ONE:
  ["analytical", "speculative", "reactive", "meme_heavy", "mixed", "off_topic"]
- "brief_summary": ≤ 150 words. Adds concrete details about THIS comment group that are NOT already captured by thread_topic, financial_angle, or conversation_character. Do not restate or rephrase those fields. Instead, add specifics found in the comments — for example: price levels discussed, named events or catalysts, specific positions or trades mentioned, dominant sub-topics, notable individual views, or any other information you find relevant and not already expressed above. Write in plain flowing text, no sub-fields or labels.

Rules:
- Be specific to THIS group, not the thread as a whole.
- Prefer concrete details over vague descriptions.
- Do not repeat the prompt or the comments verbatim.
- No markdown, no explanations outside the JSON.
- Keep the JSON valid and compact.

SUBMISSION TEXT:
{submission_text}

CURRENT COMMENT GROUP:
{chunk_comments}

Return exactly this JSON structure:
{{"main_stock_or_company":"...","thread_topic":"...","financial_angle":"...","conversation_character":"...","brief_summary":"..."}}
""".strip()


def build_chunk_summary_with_context_prompt(
    submission_text: str,
    previous_chunk_summary_json: str,
    chunk_comments: str,
) -> str:
    """
    Prompt for chunk N > 0 (previous chunk summary available as context).

    Generates a summary specific to THIS chunk. The previous chunk summary is
    provided only to help interpret references and conversational continuity —
    it is NOT a base to update. The output must describe what is happening in
    the current comment group, not in the thread as a whole.
    """
    return f"""
You are assisting an academic NLP project on Reddit discussions about stocks.

Your task is to summarize a specific group of comments from a Reddit thread.
You are NOT classifying individual comments.
You are NOT deciding whether comments are relevant or not relevant.

You are given:
1. the submission text (thread context)
2. the summary of the PREVIOUS comment group (to resolve references and understand continuity)
3. the CURRENT comment group to summarize

Return exactly ONE JSON object and nothing else.

Field definitions:
- "main_stock_or_company": stocks/companies explicitly discussed in THIS group, pipe-separated. Use "unclear" only if none can be identified.
- "thread_topic": one sentence on what the submission is about.
- "financial_angle": dominant financial angle of THIS group. Choose EXACTLY ONE:
  ["price_action", "trading_positions", "earnings", "valuation", "company_news", "sector_macro", "product_or_company_discussion", "mixed", "unclear"]
- "conversation_character": dominant tone of THIS group. Choose EXACTLY ONE:
  ["analytical", "speculative", "reactive", "meme_heavy", "mixed", "off_topic"]
- "brief_summary": ≤ 150 words. Adds concrete details about THIS comment group that are NOT already captured by thread_topic, financial_angle, or conversation_character, and NOT already stated in the previous group's summary. Do not restate or rephrase those fields or the previous summary. Instead, add specifics found in the comments — for example: price levels discussed, named events or catalysts, specific positions or trades mentioned, dominant sub-topics, notable individual views, or any other information you find relevant and not already expressed above. If the topic or tone has shifted from the previous group, briefly note what changed. Write in plain flowing text, no sub-fields or labels.

Rules:
- Describe THIS group, not the thread as a whole.
- Do not restate the previous summary.
- Prefer concrete details over vague descriptions.
- Do not repeat the prompt or the comments verbatim.
- No markdown, no explanations outside the JSON.
- Keep the JSON valid and compact.

SUBMISSION TEXT:
{submission_text}

PREVIOUS COMMENT GROUP SUMMARY (context only — do not restate):
{previous_chunk_summary_json}

CURRENT COMMENT GROUP:
{chunk_comments}

Return exactly this JSON structure:
{{"main_stock_or_company":"...","thread_topic":"...","financial_angle":"...","conversation_character":"...","brief_summary":"..."}}
""".strip()
