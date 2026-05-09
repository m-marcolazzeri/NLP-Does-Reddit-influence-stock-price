#!/bin/bash
#SBATCH --job-name=lda_pipeline
#SBATCH --account=3184584
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/home/3184584/Reddit-tech-stocks-NLP/logs/pipeline_%j.log
#SBATCH --error=/home/3184584/Reddit-tech-stocks-NLP/logs/pipeline_%j.err

set -euo pipefail

module purge
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate qwen7b

cd /home/3184584/Reddit-tech-stocks-NLP
mkdir -p logs data/topic_modeling

# Runs Step 1 (build corpus) and Step 2 (K search) only.
# After this job completes:
#   1. Download coherence_scores.csv and coherence_plot.png from HPC.
#   2. Open notebooks/06_lda_inspection.ipynb and choose K.
#   3. Set K_FINAL in src/topic_modeling/config_lda.py.
#   4. Upload config_lda.py to HPC.
#   5. Submit Step 3: sbatch run_train_lda.sh

echo "[$(date)] Step 1 - build corpus"
python src/topic_modeling/01_build_corpus.py

echo "[$(date)] Step 2 - search K"
python src/topic_modeling/02_search_k.py

echo "[$(date)] Steps 1+2 complete. Human K-selection required before running Step 3."