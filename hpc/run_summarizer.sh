#!/bin/bash
#SBATCH --job-name=summarizer
#SBATCH --account=3184584
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/3184584/Reddit-tech-stocks-NLP/logs/summarizer_%j.log
#SBATCH --error=/home/3184584/Reddit-tech-stocks-NLP/logs/summarizer_%j.err

set -euo pipefail

module purge
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate qwen7b

cd /home/3184584/Reddit-tech-stocks-NLP
mkdir -p logs data/summarization

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "[$(date)] Starting thread summarizer"
python src/summarization/run_thread_summarizer.py
echo "[$(date)] Summarizer complete"
