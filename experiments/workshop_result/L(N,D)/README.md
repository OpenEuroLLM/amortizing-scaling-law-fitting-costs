# L(N,D) — parameter/data recovery (placeholder)

**Placeholder** for the L(N,D) arm of the study. Form **`chinchilla3`** —
`L(N,D) = E + A·N^(-α) + B·D^(-β)` — following the colleague's `cwindow_*` recipe. Run to full
convergence, **10 seeds** per arm.

Two arms, differing only in the acquisition compute window:

| arm | filename | `--C_window` |
|-----|----------|--------------|
| **full**   | `cwindow_full`           | `1e-5 1e3` (no window) |
| **window** | `cwindow_curriculum_bwd` | `1e-5 1.5` (wide backward reach, tight forward cap) |

Coefficients recovered: `A, α, B, β, E` (the analysis grid shows `α, β, E` by default).

## ⚠ Not the same recipe as L(C)

This recipe differs from `../L(C)` in several settings, so **the two grid rows are not a controlled
comparison** — each is an internally-valid full-vs-window study, but form is confounded with these knobs:

| knob | L(C) | L(N,D) |
|------|------|--------|
| selection | `ei` | `random` |
| split axis | `C` | `N` |
| held-out frac | 0.5 | 0.1 |
| init compute factor | 1 | 10 |

On `oellm_english`, the `N` / 0.1 split gives **pool = 850**, **held-out = 92**, envelope set = 20,
and **6** held-out points on the global frontier (vs L(C)'s pool 905 / held-out 37 / 2 held-frontier).
The analysis notebook computes each form's split, arms, and frontier **independently**, so these
settings are handled correctly when the runs are dropped in. To make the rows directly comparable,
match these knobs to L(C).

## Reproduce

```bash
bash "experiments/workshop_result/L(N,D)/run.sh"
```

Writes to `experiments/workshop_result/L(N,D)/runs/` (gitignored). Read as the **bottom** row of
`../analysis/recovery_grid.ipynb`; until the runs exist it renders as a labelled placeholder.
