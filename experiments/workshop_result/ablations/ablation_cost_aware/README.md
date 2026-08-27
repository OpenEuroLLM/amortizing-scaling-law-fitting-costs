# ablation_cost_aware

**Ablation E-cost** — windowing vs cost-weighting on `oellm_english` / `L0`, GP `matern`, practice-style
init, run to convergence, **10 seeds**. Does restricting the candidate pool (a compute window) beat
down-weighting expensive points (a cost penalty) under full EI?

## Arms

| arm (filename) | `--selection` | acquisition | `--C_window` |
|-----|----------------|-------------|--------------|
| `ei_full` | `ei` | EI | `1e-5 1e3` (no window) |
| `ei_window` (ours) | `ei` | EI | `1e-5 1.5` (window) |
| `ei_logcost` | `ei_per_cost` | `EI / log(cost)` — **soft** penalty | `1e-5 1e3` |
| `ei_hardcost` | `ei_inv_cost` | `EI / cost` — **hard** (true inverse) penalty | `1e-5 1e3` |

`cost = C = 6·N·D` (FLOPs). Both cost-aware selectors log-transform features before the GP. Common:
`--model_type gp --gp_kernel matern` + the shared init/fit settings.

## Plots — `ablation_cost_aware_plots.ipynb`

- **(a)** coefficient recovery, one curve per arm → `out/fig_abl_cost_regret_*.pdf`.
- **(b)** observed envelope recovery → `out/fig_abl_cost_envelope_recovery.pdf`.
- **(c)** frontier map, one panel per arm → `out/fig_abl_cost_frontier_map.pdf`.

Headline: does the window beat **both** cost-weighting schemes at equal cumulative compute, and is the
soft `EI/log(cost)` penalty ≈ plain EI (log(cost) is nearly flat across the pool)?

## Runs

Google Drive folder [`11gZ…`](https://drive.google.com/drive/folders/11gZkWIh4GYBjE9mdTFvLHBXU1YJdBH5Z) → **`ablation_cost_aware.zip`** (unzips to `ablation_cost_aware/`: `runs/<arm>/…/seed_*/` + `run.sh` + plotting notebook). The analysis notebook pulls it with `ablation_fetch.fetch("cost_aware")` (id in `../../analysis/ablations_manifest.json`). Each seed ≈ 40 min CPU.
