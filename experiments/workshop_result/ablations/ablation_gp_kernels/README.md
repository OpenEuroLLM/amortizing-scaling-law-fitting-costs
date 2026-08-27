# ablation_gp_kernels

**Ablation C** — GP kernel choice on `oellm_english` / `L0`, window BO+EI (`--C_window 1e-5 1.5`),
practice-style init, run to convergence, **10 seeds**.

## Arms

Sweep `--gp_kernel` over the four registered kernels (`core/surrogates/gp.py:KERNEL_NAMES`), plus a
`vanilla` full-pool reference:

| arm (filename) | `--gp_kernel` | `--C_window` |
|-----|----------------|--------------|
| `k_matern` | `matern` (Matern-5/2, ARD) | `1e-5 1.5` |
| `k_linear` | `linear` | `1e-5 1.5` |
| `k_linmat` | `linear+matern` | `1e-5 1.5` |
| `k_lnd_mathp` | `linear_nd+matern_hp` (additive: linear on N,D + Matern on HPs) | `1e-5 1.5` |
| `vanilla` | `matern` | `1e-5 1e3` (no window) |

Common: `--model_type gp --selection ei` + the shared init/fit settings.

## Plots — `ablation_gp_kernels_plots.ipynb`

- **headline metric** — mean ± std incumbent regret (`α`) at 25% / 50% of full-pool compute, and the
  **AUC** of the observed envelope-recovery curve, as bar plots per kernel →
  `out/fig_abl_kernels_headline_metric.pdf`.
- **(a)/(b)/(c)** the full regret curves, envelope recovery, and frontier map (as in the other
  ablations) → `out/fig_abl_kernels_*.pdf`.

## Runs

Google Drive folder [`11gZ…`](https://drive.google.com/drive/folders/11gZkWIh4GYBjE9mdTFvLHBXU1YJdBH5Z) → **`ablation_gp_kernels.zip`** (unzips to `ablation_gp_kernels/`: `runs/<arm>/…/seed_*/` + `run.sh + run_full_extra.sh` + plotting notebook). The analysis notebook pulls it with `ablation_fetch.fetch("gp_kernels")` (id in `../../analysis/ablations_manifest.json`). Each seed ≈ 40 min CPU.
