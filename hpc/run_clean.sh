#!/bin/bash
#SBATCH --job-name=nlp_clean
#SBATCH --account=3184584
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3184584/Reddit-tech-stocks-NLP/logs/clean_%j.log
#SBATCH --error=/home/3184584/Reddit-tech-stocks-NLP/logs/clean_%j.err

module purge
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate qwen7b

cd /home/3184584/Reddit-tech-stocks-NLP
mkdir -p logs

echo "[$(date)] Starting clean_text.py"
python src/corpus_building/clean_text.py
echo "[$(date)] Done."
