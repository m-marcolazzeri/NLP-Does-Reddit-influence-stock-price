#!/bin/bash
#SBATCH --job-name=lda_train
#SBATCH --account=3184584
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3184584/Reddit-tech-stocks-NLP/logs/train_lda_%j.log
#SBATCH --error=/home/3184584/Reddit-tech-stocks-NLP/logs/train_lda_%j.err

set -euo pipefail

module purge
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate qwen7b

cd /home/3184584/Reddit-tech-stocks-NLP
mkdir -p logs data/topic_modeling

echo "[$(date)] Step 3 - train final LDA (K_FINAL from config_lda.py)"
python src/topic_modeling/03_train_lda.py
echo "[$(date)] Step 3 complete"
