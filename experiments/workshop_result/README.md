# workshop_result

Recovery study across two loss forms, each as a full-vs-window BO experiment, presented together as a
single two-row grid.

```
workshop_result/
├── L(C)/          # form L0    — L(C) = L0 + A·C^α        (generation scripts; runs/ gitignored)
├── L(N,D)/        # form chinchilla3 — L(N,D) = E + A/N^α + B/D^β   (placeholder generation scripts)
└── analysis/
    └── recovery_grid.ipynb    # shared notebook → out/fig_recovery_grid.pdf
```

## The grid

`analysis/recovery_grid.ipynb` draws **two rows**, `L(C)` on top and `L(N,D)` on the bottom. Each row is
the `bo_ei_recovery`-style strip: coefficient recovery `|θ̂ − θ^search|` for a few coefficients, held-out
frontier MSE `MSE_Π(G_held)(L̂_θ)`, and Envelope Set Coverage — both arms overlaid (**full** = whole pool
every step, **window** = growing slab), solid = *obs + fantasized* / dashed = *observed*, IQR bands over
seeds, one shared legend below.

Each form is configured **independently** in the notebook's `FORMS` dict — its own `run_dir`, `arms`
(filenames), coefficients, and pool/held-out split (`split_axis`, `held_out_frac`). A form with no runs
renders as a labelled placeholder, so the grid builds before the L(N,D) runs exist.

## ⚠ The two rows are not a controlled comparison

The forms currently use different recipes, so each row is an internally-valid full-vs-window study, but
"L(C) vs L(N,D)" is confounded with selection + split + init. Full config diff (dataset `oellm_english`,
seeds 0–9 both):

| setting | L(C) | L(N,D) |
|---|---|---|
| `--form` | `L0` | `chinchilla3` |
| `--selection` | `ei` | `random` |
| `--split_axis` | `C` | `N` |
| `--held_out_frac` | `0.5` | `0.1` |
| `--init_compute_factor` | `1` | `10` |
| `--init_practice` | on (`--init_practice_ds 4`) | off |
| arm filenames (full / window) | `bo_ei_full` / `bo_ei` | `cwindow_full` / `cwindow_curriculum_bwd` |
| `--model_type` | `gp` | `gp` |
| `--niters` | `1000` | `1000` |
| `--init_points` | `10` | `10` |
| `--fit_quality` | `dense` | `dense` |
| `--feature_set` | `default` | `default` |
| window factors (full / window) | `1e-5 1e3` / `1e-5 1.5` | `1e-5 1e3` / `1e-5 1.5` |
| → pool / held-out (consequence) | 905 / 37 | 850 / 92 |
| → envelope / held-frontier pts | 21 / 2 | 20 / 6 |

Six knobs differ (form, selection, split axis, held-out frac, init compute factor, init-practice) plus
arm naming; everything else matches. To make the rows directly comparable, align those knobs to L(C).
See `L(N,D)/README.md` for how.

## Reproduce

```bash
bash "experiments/workshop_result/L(C)/run.sh"      # top row
bash "experiments/workshop_result/L(N,D)/run.sh"    # bottom row (placeholder recipe)
# then run analysis/recovery_grid.ipynb
```
