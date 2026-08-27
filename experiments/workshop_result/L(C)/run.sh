#!/usr/bin/env bash
# Reproduce the L(C) recovery runs: BO with Expected Improvement on oellm_english / L0 / GP,
# practice-style init, run to full convergence (whole acquirable pool acquired), 10 seeds per arm.
# L(C) = L0 + A*C^alpha. Two arms differ only in the acquisition compute window [lo*C_lim, hi*C_lim]:
#   window (bo_ei)      -- C_window 1e-5 1.5  (ceiling grows with the frontier)
#   full   (bo_ei_full) -- C_window 1e-5 1e3  (ceiling clears the whole pool every step)
#
# Run from the repo root:  bash "experiments/workshop_result/L(C)/run.sh"
# runs/ is gitignored; each seed is ~40 min CPU to full convergence.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

DATASET=oellm_english
INPUT=data/$DATASET/preprocessed/$DATASET.csv
OUTDIR="experiments/workshop_result/L(C)/runs"
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# shared settings for both arms
COMMON=(
  --input "$INPUT"
  --output_dir "$OUTDIR"
  --form L0 --model_type gp --selection ei --niters 1000
  --init_points 10 --init_practice --init_practice_ds 4 --init_compute_factor 1
  --fit_quality dense --split_axis C --held_out_frac 0.5 --feature_set default
)

# arm: filename  C_window-lo  C_window-hi
run_arm () {
  local filename=$1 lo=$2 hi=$3
  for s in "${SEEDS[@]}"; do
    echo "=== $filename seed $s (C_window $lo $hi) ==="
    KMP_DUPLICATE_LIB_OK=TRUE python main.py loss_law \
      "${COMMON[@]}" --filename "$filename" --seed "$s" --C_window "$lo" "$hi"
  done
}

run_arm bo_ei      1e-5 1.5    # window arm (growing slab)
run_arm bo_ei_full 1e-5 1e3    # full arm  (whole pool available every step)

echo "done. open experiments/workshop_result/analysis/recovery_grid.ipynb and run all to regenerate the grid."
