#!/bin/bash
#SBATCH --job-name=lda_corpus
#SBATCH --account=3184584
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3184584/Reddit-tech-stocks-NLP/logs/corpus_%j.log
#SBATCH --error=/home/3184584/Reddit-tech-stocks-NLP/logs/corpus_%j.err

set -euo pipefail

module purge
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate qwen7b

cd /home/3184584/Reddit-tech-stocks-NLP
mkdir -p logs data/topic_modeling

echo "[$(date)] Step 1 - build corpus"
python src/topic_modeling/01_build_corpus.py
echo "[$(date)] Step 1 complete"
