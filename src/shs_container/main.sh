#!/bin/bash
#SBATCH --output=src/logs/slurm-%j.out  # Standard output and error log
#SBATCH --error=src/errors/slurm-%j.err
#SBATCH --partition=veu
#SBATCH --job-name=whisper_finetune    # Job name
#SBATCH --cpus-per-task=4                  # Run on 4 CPU
#SBATCH --mem=32G                    # Job memory request
#SBATCH --gres=gpu:5

module load singularity/3.6.4
singularity exec /gpfs/projects/bsc88/singularity-images/whisper_catala python src/main.py

