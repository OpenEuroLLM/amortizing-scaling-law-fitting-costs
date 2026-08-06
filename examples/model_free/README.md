# Baseline: model-free recovery (`--model_free`)

A **fully model-free** baseline: sample points (e.g. uniformly at random) and fit `L(C)=L0+A·C^α`
**directly on the observed compute-optimal points** — no surrogate anywhere. The naive "just run
some configs and fit the law" approach; the reference against which surrogate-based recovery should
justify itself.

## How it plugs in

The driver (`iterative_fits_L0_form.py`) normally fits a GP each iteration to infill the unobserved
grid before the law fit. `--model_free` skips that: it fits the law straight on the acquired points.
It's a **recovery** mode, orthogonal to selection.

| file | change |
|---|---|
| `iterative_fits_L0_form.py` | `--model_free` flag; when set, `build_fit` runs on the observed points and the GP infill is skipped (`heldout_mse` becomes `NaN`). Guards against surrogate-based selectors (see below). |

**Compatibility.** `--model_free` works with any *surrogate-free* selector (`random`,
`local_search`, `farthest_point`). It is **rejected** for selectors that need a GP to *pick*
(anything with `needs_surrogate = True`, e.g. `max_variance`) — those can't run without a model, so
the driver raises a clear error.

## Run it

```bash
# plain model-free random baseline
uv run python iterative_fits_L0_form.py \
    --input data/steplaw_clean.csv --output_dir results/ \
    --seed 0 --niters 24 --init_points 10 --selection random --model_free

# does the GP-infill help? model-free vs GP-infill recovery on identical random picks
uv run python examples/model_free/compare_random.py --seeds 0-9
```

`compare_random.py` runs `random` twice per seed — with GP infill and with `--model_free` — so the
only difference is the recovery route. It draws the usual 6-panel figure (recovered law + held-out
points, held-out recovery, exponent error, and the three coefficients vs. full-grid ground truth),
written to `results/baselines/model_free/model_free_vs_gp.png` (gitignored `results/`).

## Data note

The driver expects `N, D, lr, gbsz, loss`. steplaw uses `bs` for batch size and int64 `N, D`
(`C=6ND` overflows to negatives); the example writes a float-cast, renamed `*_clean.csv` and feeds
that. Do the same for any raw steplaw grid.
