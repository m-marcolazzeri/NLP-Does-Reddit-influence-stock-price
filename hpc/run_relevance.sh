#!/bin/bash
#SBATCH --job-name=relevance
#SBATCH --account=3184584
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/3184584/Reddit-tech-stocks-NLP/logs/relevance_%j.log
#SBATCH --error=/home/3184584/Reddit-tech-stocks-NLP/logs/relevance_%j.err

# NOTE: facebook/bart-large-mnli must be cached on the HPC before submitting.
# If not yet cached, run once on the login node (which has internet access):
#   python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

set -euo pipefail

module purge
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate qwen7b

cd /home/3184584/Reddit-tech-stocks-NLP
mkdir -p logs data/relevance

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "[$(date)] Starting relevance classifier"
python src/relevance/predict_relevance_zeroshot.py
echo "[$(date)] Relevance classifier complete"
