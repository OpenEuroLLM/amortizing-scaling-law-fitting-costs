# ablation_acquisition

**Ablation D** — acquisition function on `oellm_english` / `L0`, GP `matern`, window (`--C_window 1e-5
1.5`), practice-style init, run to convergence, **10 seeds**.

## Arms

Sweep `--selection` over the surrogate-based acquisitions, plus a `vanilla` full-pool (EI) reference:

| arm (filename) | `--selection` | `--C_window` | notes |
|-----|----------------|--------------|-------|
| `af_ei` | `ei` | `1e-5 1.5` | Expected Improvement |
| `af_pi` | `pi` | `1e-5 1.5` | Probability of Improvement (greedier) |
| `af_lcb` | `lcb` | `1e-5 1.5` | Lower Confidence Bound (`--lcb_kappa 2.0`) |
| `vanilla` | `ei` | `1e-5 1e3` | no-window reference |

Common: `--model_type gp --gp_kernel matern` + the shared init/fit settings.

## Plots — `ablation_acquisition_plots.ipynb`

- **(a)** coefficient recovery — regret `|θ̂ − θ_val|` per `{L0, A, α}`, one curve per acquisition →
  `out/fig_abl_acq_regret_*.pdf`.
- **(b)** observed envelope recovery → `out/fig_abl_acq_envelope_recovery.pdf`.
- **(c)** frontier map, one panel per acquisition → `out/fig_abl_acq_frontier_map.pdf`.

## Runs

Google Drive folder [`11gZ…`](https://drive.google.com/drive/folders/11gZkWIh4GYBjE9mdTFvLHBXU1YJdBH5Z) → **`ablation_acquisition.zip`** (unzips to `ablation_acquisition/`: `runs/<arm>/…/seed_*/` + `run.sh + run_full_extra.sh` + plotting notebook). The analysis notebook pulls it with `ablation_fetch.fetch("acquisition")` (id in `../../analysis/ablations_manifest.json`). Each seed ≈ 40 min CPU.
