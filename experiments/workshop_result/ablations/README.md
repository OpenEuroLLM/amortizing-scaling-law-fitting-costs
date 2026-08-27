# ablations

The workshop ablation study, one folder per ablation. **Each folder here holds only a `README.md`** —
the arms, recipe, and the plots the ablation produces. The heavy run artifacts (per-seed
`history.parquet` / `ref_fits.json` / `surrogate_predictions.parquet`, plus the generation scripts and
plotting notebooks) are **not** in this repo; they live on Google Drive.

## Runs (Google Drive)

📁 **https://drive.google.com/drive/folders/11gZkWIh4GYBjE9mdTFvLHBXU1YJdBH5Z**

**One zip per ablation** (`ablation_<name>.zip`), so a single ablation can be fetched without pulling the
whole study. Each zip unpacks to:

```
ablation_<name>/
├── README.md                     # source of the summary reproduced here
├── run.sh (+ extra run_*.sh)      # generation script(s)
├── <ablation_name>_plots.ipynb    # original plotting notebook
└── runs/<arm>/<dataset>/<form>/seed_*/<ts>/   # per-seed sidecars
```

The analysis notebook `../analysis/ablation_analysis.ipynb` fetches each zip **by Drive file id** via
`../analysis/ablation_fetch.py`; the ids live in `../analysis/ablations_manifest.json`.

## The ablations

| folder | study | what it sweeps |
|--------|-------|----------------|
| [`ablation_datasets`](ablation_datasets/)             | A       | BO+EI window generalization across datasets (warmstart, steplaw-L0, multilingual, porian) |
| [`ablation_baselines`](ablation_baselines/)           | B       | method families: RS vs GP+EI vs TabPFN-BO vs XGBoost-ens |
| [`ablation_gp_kernels`](ablation_gp_kernels/)         | C       | GP kernel choice (matern / linear / linear+matern / linear_nd+matern_hp) |
| [`ablation_acquisition`](ablation_acquisition/)       | D       | acquisition function (EI / PI / LCB) |
| [`ablation_window_growth`](ablation_window_growth/)   | E-window| growing vs sliding compute window |
| [`ablation_cost_aware`](ablation_cost_aware/)         | E-cost  | windowing vs cost-weighting (EI/log(cost), EI/cost) |

Unless noted, every ablation is on `oellm_english` / form `L0`, GP `matern`, practice-style init, run to
convergence, **10 seeds**, with a `vanilla` full-pool (no-window) reference arm.
