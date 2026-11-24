#!/bin/bash
#SBATCH --job-name=langevin-ngog
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_%x.out

cd "$SLURM_SUBMIT_DIR"
uv run python -m visualise.langevin_sample \
    runs_scaling/plane_i2_d64_b4_prei_lr0.003_seed0 \
    --steps 3000 --chains 1024 --rec-every 25 \
    --eta 2e-4 --beta0 2 --beta-max 300 --gamma 20 --barrier 5
