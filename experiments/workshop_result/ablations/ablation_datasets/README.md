# ablation_datasets

**Ablation A** — BO+EI window generalization across datasets, `L0` / GP `matern` + `ei`, practice-style
init, run to convergence, **10 seeds**. The window arm (+ a `vanilla` full-pool reference) on each
dataset; the window's upper factor is per-dataset.

## Datasets & arms

| dataset | `--input` | window arm | vanilla |
|---------|-----------|-----------|---------|
| warmstart | `data/warmstart/preprocessed/warmstart.csv` | `1e-5 2` | `1e-5 1e3` |
| steplaw (L0 loss) | `data/steplaw/preprocessed/steplaw_dense.csv` | `1e-5 2` | `1e-5 1e3` |
| oellm_multilingual | `data/oellm_multilingual/preprocessed/oellm_multilingual.csv` | `1e-5 1.5` | `1e-5 1e3` |
| **porian** (sub-exp) | `data/porian/preprocessed/porian_main_stable_rw.csv` | sweep: `full`=`1e-5 1e3`, `w15`=`1e-5 1.5`, `w20`=`1e-5 2`, `slab`=`0.5 2` | — |

Porian is a free window sweep (nothing worked well before). "Steplaw" here means the **loss law** (`L0`),
not the optimal-HP step law (a separate `hp_law` task). Non-`oellm_english` datasets must be preprocessed
first (`python -m preprocessing.preprocess --sources warmstart steplaw_dense oellm_multilingual
porian_main_stable_rw`); the generation `run.sh` skips any dataset whose CSV is missing.

## Plots — `ablation_datasets_plots.ipynb`

Per dataset present in `runs/`: coefficient regret `|θ̂ − θ_val|` for `{L0, A, α}` and observed envelope
recovery, median + IQR over seeds → `out/fig_abl_datasets_<dataset>_*.pdf`.

## Runs

Google Drive folder [`11gZ…`](https://drive.google.com/drive/folders/11gZkWIh4GYBjE9mdTFvLHBXU1YJdBH5Z) → **`ablation_datasets.zip`** (unzips to `ablation_datasets/`: `runs/<arm>/…/seed_*/` + `run.sh + run_*.sh` + plotting notebook). The analysis notebook pulls it with `ablation_fetch.fetch("datasets")` (id in `../../analysis/ablations_manifest.json`). Each seed ≈ 40 min CPU.
