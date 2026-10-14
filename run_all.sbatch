#!/bin/bash
#SBATCH --job-name=manifold-all
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# One GPU: the sphere sweep and the plane (flat control) sweep, then tables + plots for each.
# Finished runs are skipped, so resubmitting after a time-out resumes where it stopped.
set -e
cd "$SLURM_SUBMIT_DIR"

for s in sphere plane; do
    echo "=== sweep: $s ==="
    SWEEP=$s uv run sweep.py
done

for s in sphere plane; do
    echo "=== analysis: $s ==="
    SWEEP=$s uv run table.py
    SWEEP=$s uv run plot.py
done
