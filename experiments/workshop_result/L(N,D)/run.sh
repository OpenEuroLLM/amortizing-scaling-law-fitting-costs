#!/usr/bin/env bash
# PLACEHOLDER — L(N,D) recovery runs (form chinchilla3: L(N,D) = E + A/N^alpha + B/D^beta).
# Mirrors the colleague's cwindow_* recipe, which differs from L(C) in several settings (see NOTE):
#   --selection random  --split_axis N  --held_out_frac 0.1  --init_compute_factor 10  (no --init_practice)
# Two arms differ only in the acquisition compute window [lo*C_lim, hi*C_lim]:
#   window (cwindow_curriculum_bwd) -- C_window 1e-5 1.5  (wide backward reach, tight forward cap)
#   full   (cwindow_full)           -- C_window 1e-5 1e3  (no window)
#
# NOTE: this recipe is NOT the same as L(C) (which uses --selection ei --split_axis C --held_out_frac 0.5).
# The two forms are therefore each an internally-valid full-vs-window study, but the rows are not a
# controlled L(C)-vs-L(N,D) comparison. To make them directly comparable, match these settings to L(C).
#
# Run from the repo root:  bash "experiments/workshop_result/L(N,D)/run.sh"
# runs/ is gitignored.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

DATASET=oellm_english
INPUT=data/$DATASET/preprocessed/$DATASET.csv
OUTDIR="experiments/workshop_result/L(N,D)/runs"
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# shared settings for both arms (form chinchilla3 = L(N,D); colleague's cwindow_* recipe)
COMMON=(
  --input "$INPUT"
  --output_dir "$OUTDIR"
  --form chinchilla3 --model_type gp --selection random --niters 1000
  --init_points 10 --init_compute_factor 10
  --fit_quality dense --split_axis N --held_out_frac 0.1 --feature_set default
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

run_arm cwindow_curriculum_bwd 1e-5 1.5    # window arm (backward-relaxed curriculum window)
run_arm cwindow_full           1e-5 1e3    # full arm  (no window)

echo "done. open experiments/workshop_result/analysis/recovery_grid.ipynb and run all to regenerate the grid."
