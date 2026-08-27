# ablation_baselines

**Ablation B** — method-family comparison on `oellm_english` / `L0`, practice-style init, run to full
convergence, **10 seeds**. Four acquisition methods in one plot.

## Arms

| arm | `--model_type` | `--selection` | `--C_window` | notes |
|-----|----------------|---------------|--------------|-------|
| RS | `none` | `random` | `1e-5 1e3` | model-free random search (full pool) |
| BO+EI (ours) | `gp` | `ei` | `1e-5 1.5` | GP + Expected Improvement, growing window |
| TabPFN-BO | `tabpfn_bo` | `tabpfn_ei` | `1e-5 1.5` | TabPFN posterior + exact EI |
| XGBoost-ens | `xgb_ens` | `ei` | `1e-5 1.5` | bagged XGBoost ensemble + EI (`OMP_NUM_THREADS=1`) |

Shared: `--init_points 10 --init_practice --init_practice_ds 4 --init_compute_factor 1 --fit_quality
dense --split_axis C --held_out_frac 0.5 --feature_set default --niters 1000`.

## Plots — `ablation_baselines_plots.ipynb`

- **(a)** coefficient recovery — incumbent regret `|θ̂ − θ_val|` for `{L0, A, α}` vs cumulative compute,
  median + IQR, one curve per arm → `out/fig_abl_baselines_regret_*.pdf`.
- **(b)** envelope recovery — % of the pool compute-optimal frontier **acquired** (observed frontier, so
  families compare fairly) vs compute → `out/fig_abl_baselines_envelope_recovery.pdf`.
- **(c)** frontier map — pool `(C, loss)` with envelope points colored by acquisition compute (seed 0),
  one panel per method → `out/fig_abl_baselines_frontier_map.pdf`.

## Runs

Google Drive folder [`11gZ…`](https://drive.google.com/drive/folders/11gZkWIh4GYBjE9mdTFvLHBXU1YJdBH5Z) → **`ablation_baselines.zip`** (unzips to `ablation_baselines/`: `runs/<arm>/…/seed_*/` + `run.sh` + plotting notebook). The analysis notebook pulls it with `ablation_fetch.fetch("baselines")` (id in `../../analysis/ablations_manifest.json`). Each seed ≈ 40 min CPU.
