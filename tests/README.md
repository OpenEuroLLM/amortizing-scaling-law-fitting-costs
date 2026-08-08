# Tests

Fast **smoke tests** for the `core` package — they check that the reorg's wiring holds (registries,
feature-set resolution, fitting, and both `main.py` tasks end-to-end), not the underlying scaling
math. The whole suite runs in a few seconds and needs **no GP backend**: the pipeline tests use the
model-free path (`--model_type none`).

Run with either runner (tests are plain `unittest.TestCase`, so both work):

```bash
python -m unittest discover -s tests      # no extra deps
uv run --extra dev pytest                 # or pytest, once the `dev` extra is installed
```

| File | Covers |
|---|---|
| `test_registries.py` | `PARAMETRIC_FORMS` / `MODELS` / `SELECTION` populate; loss-vs-HP form partition; `NullSurrogate` predicts NaN |
| `test_feature_sets.py` | `resolve_grid_features` drops absent/all-null/constant columns; `arch`/`data`/`full` supersets |
| `test_fitting.py` | `build_fit` returns finite, in-bounds params for `L0` and `steplaw_lr` |
| `test_pipelines.py` | `main.py loss_law` / `hp_law` run to a valid `history.parquet` |
| `_grid.py` | shared tiny synthetic `N x D x lr x gbsz` fixture grid |
