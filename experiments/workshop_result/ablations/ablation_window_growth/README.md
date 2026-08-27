# ablation_window_growth

**Ablation E-window** — sliding vs growing compute window on `oellm_english` / `L0`, GP `matern` + `ei`,
practice-style init, run to convergence, **10 seeds**. Motivates *why* the window should grow from ~0
rather than slide.

## Arms

The window is `[lo·C_lim, hi·C_lim]` with `C_lim` = max acquired compute (grows each iter,
`core/utils/data.py:next_batch_compute`). The arms differ only in the **lower** factor:

| arm (filename) | `--C_window` | behavior |
|-----|--------------|----------|
| `growing` | `1e-5 1.5` | lower bound ≈ 0 → cheap points stay eligible as the frontier grows |
| `sliding` | `0.5 1.5` | lower bound `0.5·C_lim` rises → slab slides up, abandons cheap points |
| `vanilla` | `1e-5 1e3` | no window (reference) |

## Plots — `ablation_window_growth_plots.ipynb`

- **(a)** coefficient recovery → `out/fig_abl_window_regret_*.pdf`.
- **(b)** observed envelope recovery → `out/fig_abl_window_envelope_recovery.pdf`.
- **acquisition pattern** — compute `C` of the point acquired at each iteration (seed 0), per arm:
  growing keeps sampling cheap points; sliding's `C` floor rises with the frontier →
  `out/fig_abl_window_acq_pattern.pdf`.
- **(c)** frontier map, one panel per arm → `out/fig_abl_window_frontier_map.pdf`.

## Runs

Google Drive folder [`11gZ…`](https://drive.google.com/drive/folders/11gZkWIh4GYBjE9mdTFvLHBXU1YJdBH5Z) → **`ablation_window_growth.zip`** (unzips to `ablation_window_growth/`: `runs/<arm>/…/seed_*/` + `run.sh` + plotting notebook). The analysis notebook pulls it with `ablation_fetch.fetch("window_growth")` (id in `../../analysis/ablations_manifest.json`). Each seed ≈ 40 min CPU.
