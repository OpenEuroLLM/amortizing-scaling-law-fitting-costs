# Baseline: `farthest_point` (space-filling / max-min sampling)

An acquisition strategy for the iterative scaling-law fit: at each step, **buy the candidate that
is farthest, in standardized log-`(N, D, lr, gbsz)` space, from everything observed so far**
(greedy max-min / farthest-point sampling). A **surrogate-free**, space-filling design — it spreads
points evenly over the grid without any model. Contrast `max_variance`, which uses GP posterior
variance; `farthest_point` is pure geometry.

## How it plugs in

The driver (`iterative_fits_L0_form.py`) passes the observed set to `next_point`, which returns the
candidate with the largest nearest-observed distance. The GP is still fit each iteration for the
**law recovery** (grid infill) — only the *pick* differs from `random`. No GP is used for the
selection itself.

| file | change |
|---|---|
| `selection.py` | `FarthestPointSelection` — `next_point` returns the max-min-distance candidate; bootstraps with random before anything is observed. |
| `iterative_fits_L0_form.py` | registers `"farthest_point"`; passes `acquired_idx` to `next_point`. |

`random`/`local_search` are untouched (their `next_point` ignores the extra `acquired_idx` kwarg).

## Run it

```bash
# through the canonical driver
uv run python iterative_fits_L0_form.py \
    --input data/steplaw_clean.csv --output_dir results/ \
    --seed 0 --niters 24 --init_points 10 --selection farthest_point

# compare against random over several seeds
uv run python examples/farthest_point/compare_vs_random.py --seeds 0-9
```

`compare_vs_random.py` runs the driver for `random` and `farthest_point` over the given seeds, then
draws a 6-panel figure (median over seeds, IQR band): the **recovered law `L(C)=L0+A·C^α`** against
the true compute-optimal frontier with the **held-out largest-N points** marked, **held-out
recovery** (`known_test_mse`), **exponent error** (`|alpha regret|`), and the three **recovered
coefficients** `L0, A, α` vs. their **full-grid ground truth** (dashed). The figure is written to
`results/baselines/farthest_point/farthest_point_vs_random.png` (under the gitignored `results/`).

## Data note

The driver expects columns `N, D, lr, gbsz, loss`. steplaw uses `bs` for batch size, and its raw
`N, D` are int64 — `C = 6ND` overflows int64 into negatives. The example writes a float-cast,
renamed copy (`*_clean.csv`) and feeds the driver that; do the same for any raw steplaw grid.

## Note on this grid

steplaw has only ~15 trainable `(N, D)` cells and ~119 `(lr, gbsz)` configs per cell, so results are
high-variance across seeds. Treat this as an integration example on real data, not a decisive
benchmark. Space-filling favors *coverage* of the scaling axes, which is what recovering `L(C)`
rewards.
