#!/bin/bash
#SBATCH --job-name=cwindow_curriculum_bwd
#SBATCH --array=0-9
#SBATCH --partition=rome
#SBATCH --account=REPLACE_ME          # your Snellius budget account
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00               # ~1.8h/seed measured locally (--fit_quality debug) + margin
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err

# Seed sweep for the cwindow_curriculum_bwd experiment (chinchilla3 loss_law), one seed per
# array task -- same flags as examples/run_cwindow_curriculum_bwd.py, run in parallel instead
# of sequentially. Seeds are fully independent (own output dir, no shared state).
#
#   mkdir -p logs
#   sbatch examples/run_cwindow_curriculum_bwd.sbatch

set -euo pipefail

REPO_DIR=REPLACE_ME/surrogate_based_scaling_laws   # path to this repo on Snellius
cd "$REPO_DIR"

SEED=$SLURM_ARRAY_TASK_ID

uv run python main.py loss_law \
    --input "$REPO_DIR/data/oellm_english/preprocessed/oellm_english.csv" \
    --output_dir "$REPO_DIR/results" \
    --niters 1000 \
    --seed "$SEED" \
    --model_type gp \
    --selection random \
    --form chinchilla3 \
    --feature_set default \
    --split_axis N \
    --held_out_frac 0.1 \
    --plot_figure \
    --color_fits_by_N \
    --C_window 1e-5 1.5 \
    --init_compute_factor 1 \
    --init_points 10 \
    --filename cwindow_curriculum_bwd
