# Baseline: `max_variance` (uncertainty sampling)

An acquisition strategy for the iterative scaling-law fit: at each step, **buy the candidate run
where the GP surrogate is least certain** (highest posterior standard deviation), instead of a
uniform random draw. Classic active-learning / uncertainty sampling.

## How it plugs in

The driver (`iterative_fits_L0_form.py`) already fits a GP surrogate every iteration to infill the
grid before fitting `L(C)`. `max_variance` reuses that same GP: the driver fits it on the current
observations **before** selection and hands it to `next_point`, which returns the pool point of
maximum posterior std. So it's the *same loop as `random`* — only the pick changes.

| file | change |
|---|---|
| `models.py` | `GPModel.predict(X, return_std=True)` — expose posterior std. |
| `selection.py` | `MaxVarianceSelection` — `next_point` returns the max-std candidate; `needs_surrogate = True`. Falls back to random until a GP is available (bootstrap). |
| `iterative_fits_L0_form.py` | registers `"max_variance"`; fits the selection GP before `next_point` and passes it in (only for `needs_surrogate` strategies). |

`random` and `local_search` are untouched (their `next_point` ignores the extra `surrogate` kwarg).

## Run it

```bash
# through the canonical driver
uv run python iterative_fits_L0_form.py \
    --input data/steplaw_clean.csv --output_dir results/ \
    --seed 0 --niters 24 --init_points 10 --selection max_variance

# compare against random over several seeds (recovery + exponent error, with IQR bands)
uv run python examples/max_variance/compare_vs_random.py --seeds 0-4
```

`compare_vs_random.py` runs the driver for `random` and `max_variance` over the given seeds, then
draws a 6-panel figure (median over seeds, IQR band): the **recovered law `L(C)=L0+A·C^α`** against
the true compute-optimal frontier with the **held-out largest-N points** marked, **held-out
recovery** (`known_test_mse`), **exponent error** (`|alpha regret|`), and the three **recovered
coefficients** `L0, A, α` vs. their **full-grid ground truth** (dashed). The figure is written to
`results/baselines/max_variance/max_variance_vs_random.png` (under the gitignored `results/`).

## Data note

The driver expects columns `N, D, lr, gbsz, loss`. steplaw uses `bs` for batch size, and its raw
`N, D` are int64 — `C = 6ND` overflows int64 into negatives. The example writes a float-cast,
renamed copy (`*_clean.csv`) and feeds the driver that; do the same for any raw steplaw grid.

## Note on this grid

steplaw has only ~15 trainable `(N, D)` cells and ~119 `(lr, gbsz)` configs per cell, so results are
high-variance across seeds and the acquisition choice matters less than on a grid with many model
sizes. Treat this as an integration example on real data, not a decisive benchmark.
