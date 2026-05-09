# `data/raw/` — original archives

Untouched Reddit dumps used as the source of truth for the entire pipeline. **Read-only**.

## Contents

Actual location: `data/raw/subreddits25/`

- `subreddits25/wallstreetbets_comments.zst` — JSONL stream of comments, Zstandard-compressed (~8 GB)
- `subreddits25/wallstreetbets_submissions.zst` — JSONL stream of submissions, Zstandard-compressed (~572 MB)

These files are not committed to git (`data/raw/` is gitignored). They come from Pushshift / Academic-torrents archives. If the folder is empty, the archives have not been copied to this machine yet.

## Format

Each line in the decompressed stream is a JSON object describing one Reddit object. The fields used downstream are:

- comments: `id`, `link_id`, `parent_id`, `created_utc`, `subreddit`, `author`, `body`, `score`, `permalink`
- submissions: `id`, `created_utc`, `subreddit`, `author`, `title`, `selftext`, `score`, `permalink`

## Producers / consumers

- Produced: externally (Pushshift / Academic-torrents).
- Consumed by: `src/extraction/extract_reddit_matches.py`, `src/extraction/recover_parent_submissions.py`.
