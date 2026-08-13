# Baseline: `random` selection — GP-infill vs. model-free recovery

Holds the **selection** fixed at `random` and varies the **recovery route** (`--model_type`): GP
grid-infill vs. model-free (`--model_type none`). The model-free arm fits `L(C)=L0+A·C^α`
**directly on the observed compute-optimal points** — no surrogate anywhere, the naive "just run
some configs and fit the law" approach — and is the reference against which surrogate-based recovery
should justify itself.

## How it plugs in

The driver (`main.py`) normally fits a GP each iteration to infill the unobserved
grid before the law fit. `--model_type none` (the `NullSurrogate`) skips that: it fits the law
straight on the acquired points. It's a **recovery** mode, orthogonal to selection — just another
value of the `--model_type` axis alongside `gp`.

| file | change |
|---|---|
| `main.py` | `--model_type none` selects the `NullSurrogate`, whose predictions are `NaN`; the driver drops those rows, so `build_fit` runs on the observed points only and `heldout_mse` becomes `NaN`. Guards against surrogate-based selectors (see below). |

**Compatibility.** `--model_type none` works with any *surrogate-free* selector (`random`,
`local_search`, `farthest_point`). It is **rejected** for selectors that need a GP to *pick*
(anything with `needs_surrogate = True`, e.g. `max_variance`) — those can't run without a model, so
the driver raises a clear error.

## Run it

```bash
# plain model-free random baseline
uv run python main.py \
    --input data/steplaw_clean.csv --output_dir results/ \
    --seed 0 --niters 24 --init_points 10 --selection random --model_free

# does the GP-infill help? model-free vs GP-infill recovery on identical random picks
uv run python examples/random/compare_random.py --seeds 0-9
```

`compare_random.py` runs `random` twice per seed — with GP infill and with `--model_type none` — so
the only difference is the recovery route. It draws the usual 6-panel figure (recovered law +
held-out points, held-out recovery, exponent error, and the three coefficients vs. full-grid ground
truth), written to `results/baselines/random/gp_vs_model_free.png` (gitignored `results/`).

## Data note

The driver expects `N, D, lr, gbsz, loss`. steplaw uses `bs` for batch size and int64 `N, D`
(`C=6ND` overflows to negatives); the example writes a float-cast, renamed `*_clean.csv` and feeds
that. Do the same for any raw steplaw grid.
