#!/bin/bash
#SBATCH --job-name=lda_search_k
#SBATCH --account=3184584
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3184584/Reddit-tech-stocks-NLP/logs/search_k_%j.log
#SBATCH --error=/home/3184584/Reddit-tech-stocks-NLP/logs/search_k_%j.err

set -euo pipefail

module purge
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate qwen7b

cd /home/3184584/Reddit-tech-stocks-NLP
mkdir -p logs data/topic_modeling

echo "[$(date)] Step 2 - search K"
python src/topic_modeling/02_search_k.py
echo "[$(date)] Step 2 complete"