# HPC Setup — One-Time Steps

**HPC:** `slogin.hpc.unibocconi.it` — user `3184584` — conda env `qwen7b`

Complete these steps once before running any SLURM job. Skip steps already done.

---

## 1. Create directory structure (on HPC)

```bash
ssh 3184584@slogin.hpc.unibocconi.it
cd ~/Reddit-tech-stocks-NLP
mkdir -p data/raw/subreddits25 data/extraction data/summarization \
          data/relevance data/corpus_building data/topic_modeling logs
```

## 2. Upload raw data — run locally (~8 GB, ~5 min)

```bash
scp data/raw/subreddits25/wallstreetbets_comments.zst \
    3184584@slogin.hpc.unibocconi.it:~/Reddit-tech-stocks-NLP/data/raw/subreddits25/
```

The submissions archive is only used in Stage 3, which runs locally — do not upload it.

## 3. Install missing Python package (on HPC)

```bash
module purge && module load /software/modules/miniconda3
eval "$(conda shell.bash hook)" && conda activate qwen7b
pip install zstandard
```

`zstandard` is in `requirements.txt` but is absent from the `qwen7b` conda environment.

## 4. Sync repo code — run locally (if HPC copy is stale)

```bash
tar --exclude="./data" --exclude="./.git" --exclude="./__pycache__" \
    -czf /tmp/redditNLP_code.tar.gz .
scp /tmp/redditNLP_code.tar.gz \
    3184584@slogin.hpc.unibocconi.it:~/redditNLP_code.tar.gz
```

```bash
# On HPC:
tar -xzf ~/redditNLP_code.tar.gz -C ~/Reddit-tech-stocks-NLP --overwrite
rm ~/redditNLP_code.tar.gz
```

## 5. Download LLM models (on login node)

**Compute nodes have no internet.** Models must be downloaded on the login node
(`slnode01`), which does have internet. Downloads go to `~/.cache/huggingface/`,
which is shared with compute nodes.

Check available disk space first:
```bash
df -h ~   # need ~32 GB free
```

**Qwen2.5-14B-Instruct — Stage 5 (~30 GB, ~15–20 min):**
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-14B-Instruct')"
```

**facebook/bart-large-mnli — Stage 6 (~1.6 GB, ~1 min):**
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/bart-large-mnli')"
```

After download, all `run_*.sh` scripts set `TRANSFORMERS_OFFLINE=1` automatically —
no further action needed.
