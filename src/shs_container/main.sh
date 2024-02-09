#!/bin/bash
#SBATCH -D .
#SBATCH --output=src/logs/slurm-%j.out  # Standard output and error log
#SBATCH --error=src/errors/slurm-%j.err
#SBATCH --job-name=whisper_finetune    # Job name
#SBATCH --cpus-per-task=1                  # Run on 4 CPU
#SBATCH --mem=32G                    # Job memory request
#SBATCH --gres=gpu:1
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
module load singularity/3.6.4
singularity exec /gpfs/projects/bsc88/singularity-images/whisper-catala python src/main.py

