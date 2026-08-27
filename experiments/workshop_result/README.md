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

The forms currently use different recipes (L(C): `ei` / split `C` / held-out 0.5; L(N,D): `random` /
split `N` / held-out 0.1). So each row is an internally-valid full-vs-window study, but "L(C) vs L(N,D)"
is confounded with selection + split. See `L(N,D)/README.md` for the table and how to align them.

## Reproduce

```bash
bash "experiments/workshop_result/L(C)/run.sh"      # top row
bash "experiments/workshop_result/L(N,D)/run.sh"    # bottom row (placeholder recipe)
# then run analysis/recovery_grid.ipynb
```
