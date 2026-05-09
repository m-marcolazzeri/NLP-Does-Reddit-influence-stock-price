#!/bin/bash
#SBATCH --job-name=nlp_chunks
#SBATCH --account=3184584
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/3184584/Reddit-tech-stocks-NLP/logs/chunks_%j.log
#SBATCH --error=/home/3184584/Reddit-tech-stocks-NLP/logs/chunks_%j.err

module purge
module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate qwen7b

cd /home/3184584/Reddit-tech-stocks-NLP
mkdir -p logs

echo "[$(date)] Starting build_chunks.py"
python src/corpus_building/build_chunks.py
echo "[$(date)] Done."
