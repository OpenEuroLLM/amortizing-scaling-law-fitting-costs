# Surrogate-Based Scaling Laws

Fits neural scaling laws (loss vs. compute) using a surrogate model trained on a subset of observed runs, rather than requiring all runs to be evaluated.

## Overview

A standard scaling law study sweeps many combinations of model size `N`, dataset size `D`, learning rate `lr`, and batch size `gbsz`, then fits a power law `L(C) = A * C^(-alpha)` to the Pareto-optimal (loss envelope) points. Running every combination is expensive. This repo instead:

1. Trains a **surrogate model** on a small sampled subset of runs (in log-space over `N`, `D`, `lr`, `gbsz`).
2. Uses the surrogate to **predict loss** for all configurations cheaply.
3. Extracts the **Pareto frontier** (best loss per compute budget `C = 6ND`) from the predictions.
4. Fits a **power law** to the envelope and reports `A` and `alpha`.

## Layout

| Path | Description |
|---|---|
| `main.py` | **Entry point** — dispatches to a task: `main.py loss_law ...` or `main.py hp_law ...` |
| `core/pipelines/` | Task pipelines: `loss_law` (loss-vs-compute/`N,D` acquisition loop) and `hp_law` (optimal-HP/step-law recovery) |
| `core/selection/` | Training-point selection strategies (`random`, `local_search`, `max_variance`, `farthest_point`) + `SELECTION` registry |
| `core/surrogates/` | Surrogate model backends: GP (BoTorch/GPyTorch), TabPFN, AutoGluon + `MODELS` registry |
| `core/forms/` | Parametric `L(...)` forms (L0, chinchilla3, chinchilla_interaction, width_depth, steplaw_lr/bsz) + `PARAMETRIC_FORMS` registry |
| `core/fitting.py` | `build_fit` — fit a parametric form to (subsampled / surrogate-infilled) data |
| `core/utils/` | Data preprocessing helpers (`data.py`) and plotting styles (`plotting.py`) |
| `extra/one_shot_fit.py` | Alternative one-shot baseline: sample once, fit a GP, fit a pure power law, dump coefficients to JSON |
| `preprocessing/fetch_data.py` | Downloads the raw CSV from each source repo into `data/raw/` |
| `preprocessing/preprocess.py` | Adapts each raw CSV into the common `N, D, lr, gbsz, loss` schema in `data/preprocessed/` |
| `submodules/EffingScaling` | Vendored `scaling` package (parametric functional forms, Pareto/fit utilities) |
| `examples/` | Per-strategy comparison scripts that drive `main.py loss_law` |

## Usage

### Setup

Install the environment with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git submodule update --init submodules/EffingScaling
uv sync --extra gp        # --extra gp pulls in botorch/gpytorch/torch for the GP surrogate
```

Quick smoke test (assuming a preprocessed CSV exists):

```bash
python main.py loss_law \
    --seed 12345 \
    --niters 25 \
    --output_dir results/iterative/ \
    --input data/all_points_clean.csv \
    --init_points 25 \
    --init_compute_factor 50 \
    --C_window 0.1 50 \
    --plot_figure
```

### Data preprocessing

```bash
uv run python preprocessing/fetch_data.py    # -> data/raw/<source>/...
uv run python preprocessing/preprocess.py    # -> data/preprocessed/<source>.csv
```

`dense_english_scaling_results` and `dense_multilingual_models_scaling_results` are private OpenEuroLLM repos. So it needs your local git to already be able to authenticate to them. Request access first if you can't clone them yourself.

| Source | Origin | Has `lr`/`gbsz`? |
|---|---|---|
| `oellm_english` | OpenEuroLLM dense English scaling results (private repo) | yes |
| `oellm_multilingual` | OpenEuroLLM dense multilingual scaling results (private repo) | yes |
| `steplaw_dense` | Li et al., arXiv:2503.04715 | yes |
| `steplaw_moe` | Li et al., arXiv:2503.04715 | yes |
| `porian` | Porian et al., arXiv:2406.19146 | yes |
| `gemstones` | McLeish et al., arXiv:2502.06857 | no |
| `activesl_lr_bsz` | active-experiment-selection benchmark, arXiv:2604.22753 | yes |
| `activesl_farseer` | active-experiment-selection benchmark, arXiv:2604.22753 | no |
| `activesl_data_constrained` | active-experiment-selection benchmark, arXiv:2604.22753 | no |
| `warmstart` | warmstarting-for-scaling-language-models runs (no public source -- see below) | yes |

Sources with no `lr`/`gbsz` in the underlying data leave those columns blank -- the `--feature_set default` GP auto-narrows to `N, D` for them (fine for `L0`/`chinchilla`), while `extra/one_shot_fit.py` still needs all four features.

`warmstart` has no automated fetch step: its raw file has no public source, so `data/raw/warmstart/warmstart_runs_flattened.parquet` must be placed manually before running `preprocess.py`.

5 sub-datasets from the active-experiment-selection benchmark (arXiv:2604.22753) aren't included, because their columns don't reduce to `N, D, lr, gbsz, loss` at all yet:

| Sub-dataset | Why it's excluded |
|---|---|
| `domain_mixture_scaling_law` | 5 mixture-proportion features and 5 per-domain losses -- no `N`/`D` concept |
| `moe_scaling_law` | `num_experts` + `dense_parameter_count` -- no `D`/`lr`/`gbsz` |
| `parallel_scaling_law` | inference-time `parallel_size` -- no `D`/`lr`/`gbsz` |
| `sparsity_scaling_law` | two separate `N`s (active/dense) and two `D`s |
| `vocab_scaling_law` | `vocab_size` + `num_characters` in place of tokens, and an already vocab-normalized loss |

### N-variant vocabulary

`N` can mean different things depending on whether the embedding and/or output head are counted. Where a preprocessed CSV has an `N_definition` column or extra `N_*` columns, this is the shared vocabulary (dataset-specific READMEs and code comments link here instead of repeating it):

| Name | Definition |
|---|---|
| `N_total` | Every trainable parameter (embedding/head counted once if tied, twice if untied). Porian et al. (arXiv:2406.19146) call this `N_exact` -- the name `porian.csv` uses for its own column. |
| `N_ne` | Excludes both the input embedding and output head/unembedding matrices ("non-embedding"). Porian et al. call this `N_Kaplan`, after Kaplan et al.'s original convention -- the name `porian.csv` uses for its own column. |
| `N_Porian` | Porian et al.'s own default: excludes the input embedding but includes the head; close to `N_total` when embeddings are tied, since the head then reuses the embedding matrix. Only appears in `porian.csv`. |
| `N_eff` | Porian et al.'s "effective" parameter count: `N_Porian` adjusted upward for attention-operation FLOPs (bigger relative effect on smaller models). Only appears in `porian.csv`. |

### `loss_law` — iterative surrogate-based loss-law fit (main pipeline)

Recovers a **loss** scaling law (`L` vs compute / `N,D`) via the acquisition loop:

```bash
python main.py loss_law \
    --input        path/to/all_points_clean.csv \
    --output_dir   results/ \
    --niters       50 \          # acquisition iterations (or use --C_budget for a compute budget)
    --seed         1 \
    --model_type   gp \          # none | gp | tabpfn | ag  ('none' = model-free)
    --selection    random \      # random | local_search | max_variance | farthest_point
    --form         L0 \          # L0 | chinchilla3 | chinchilla_interaction | width_depth
    --feature_set  default \     # default | arch | data | full  (which features the GP sees)
    --split_axis   C \           # C | N  (axis the fraction split runs along)
    --held_out_frac 0.2          # top fraction by --split_axis held out (extrapolation eval)
```

Each iteration selects a new point, (optionally) infills the rest of the grid with the surrogate, fits the chosen parametric `--form`, and logs the fit, per-parameter regret, and the fitted law's MSE on **both regions**. The run writes `output_dir/seed_{seed}/history.parquet` (add `--plot_figure` for a diagnostic PNG). Use `--model_type none` (model-free) to fit directly on the observed points and skip the surrogate infill.

**Two-region evaluation.** The grid is split along `--split_axis` into the acquire-able **pool** (everything below the threshold — the loop draws from it) and the top-fraction **held-out** region (`--held_out_frac` — the largest scales, never acquired). Every iteration scores the recovered law on both, written as `law_mse_pool` (in-region recovery on the searched scales) and `law_mse_held_out` (extrapolation; `known_test_mse` is kept as a back-compat alias of the latter). Their gap is the in-region→held-out generalization gap — the diagnostic/comparison figures plot the two as a solid (pool) + dashed (held-out) pair.

`--form` here covers the **loss** forms. L-vs-C forms (`L0`, `chinchilla3`, `chinchilla_interaction`) are scored on each region's compute-optimal frontier and extrapolated beyond the grid; `width_depth` has no compute axis, so it's scored by **direct loss MSE** at the config points (pair it with a `--feature_set` that exposes `width`/`depth`). The optimal-HP forms (`steplaw_lr`, `steplaw_bsz`) are a different task — see `hp_law` below.

**GP feature set (`--feature_set`).** The GP always predicts `loss`; this flag chooses which columns it conditions on, to test whether richer features improve loss-infill and law recovery:

| mode | GP features |
|---|---|
| `default` | `N, D, lr, gbsz` |
| `arch` | `default` + architectural HPs (`width, depth, ffn, num_heads`) |
| `data` | `default` + data-axis HPs (`unique_tokens, seq_len`) |
| `full` | `default` + both |

`arch`/`data` isolate each group's contribution. Every candidate column is auto-narrowed to those **present, non-null, and non-constant**, so datasets missing a knob (HP-less sources with all-`NaN` `lr`/`gbsz`, or CSVs not built with `preprocess.py --extended`) simply drop it — the extra modes then collapse toward `default`. The architecture/data columns only exist under `preprocess.py --extended`, so the non-`default` modes are only meaningful on datasets that carry that resolution (e.g. `steplaw`, `gemstones`).

### `hp_law` — optimal-HP (step-law) recovery

The step-law forms fit the **optimal hyperparameter** as a function of scale, not loss — so they get their own task:

```bash
python main.py hp_law \
    --input        path/to/steplaw.csv \
    --output_dir   results/ \
    --form         steplaw_lr \   # steplaw_lr (optimal lr vs N,D) | steplaw_bsz (optimal batch size vs D)
    --held_out_frac 0.2 \         # largest fraction of scales held out (extrapolation eval)
    --plot_figure
```

It reduces the grid to the optimal HP **per scale** (min-loss row at each `N,D` / `D`) and fits the step law to those optima. It uses the same **two-region** split as `loss_law`, but along the form's scale axis: the largest `--held_out_frac` of scales are held out (extrapolation) and the rest are the acquire-able pool (needs ≥3 scales to fit). Each iteration logs the predicted-vs-actual optimal-HP log-MSE on both — `law_mse_pool` (in-region) and `law_mse_held_out`. Writes `output_dir/seed_{seed}/history.parquet`.

### One-shot surrogate fit (alternative baseline)

```bash
python -m extra.one_shot_fit \
    --input      path/to/all_points_clean.csv \
    --output_dir results/ \
    --num_points 50 \          # absolute number of training runs
    --seed       1 \
    --model_type gp \          # gp | tabpfn | ag
    --selection  random        # random | local_search | max_variance | farthest_point
```

Results are written to `output_dir` as `results_{model_type}_{selection}_{num_points}_points_seed_{seed}.json`. Each file contains the fitted `A_fit`, `alpha_fit`, surrogate MSE, number of training points, and total compute of the training subset.

## Input data format

A CSV with at least these columns:

| Column | Description |
|---|---|
| `N` | Model parameter count |
| `D` | Number of training tokens |
| `lr` | Learning rate |
| `gbsz` | Global batch size |
| `loss` | Validation loss |

## Surrogate models

| Key | Class | Backend |
|---|---|---|
| `none` | `NullSurrogate` | No surrogate (model-free): fit `L(C)` directly on the observed compute-optimal points |
| `gp` | `GPModel` | BoTorch + GPyTorch (Matérn kernel, ARD) |
| `tabpfn` | `TabPFNModel` | TabPFN |
| `ag` | `AutoGluonModel` | AutoGluon TabularPredictor |

All models receive log-transformed features.

## Selection strategies

| Key | Class | Description |
|---|---|---|
| `random` | `RandomSelection` | Uniformly samples runs at random |
| `local_search` | `LocalSearch` | Starts at the smallest scale and performs coordinate-wise local search over `lr`/`gbsz`, transferring the incumbent to larger `N`/`D` scales |
| `max_variance` | `MaxVarianceSelection` | Uncertainty sampling: acquires the pool point of highest GP posterior variance (needs the surrogate) |
| `farthest_point` | `FarthestPointSelection` | Max-min space-filling: acquires the point farthest from everything observed, in standardized log-feature space |

## Dependencies

Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

Per surrogate backend:
- GP: `botorch`, `gpytorch`, `torch`
- TabPFN: `tabpfn`
- AutoGluon: `autogluon.tabular`
