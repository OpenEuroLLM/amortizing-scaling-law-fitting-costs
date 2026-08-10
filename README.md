# Surrogate-Based Scaling Laws

Fits neural scaling laws (loss vs. compute) using a surrogate model trained on a subset of observed runs, rather than requiring all runs to be evaluated.

## Overview

A standard scaling law study sweeps many combinations of model size `N`, dataset size `D`, learning rate `lr`, and batch size `gbsz`, then fits a power law `L(C) = A * C^(-alpha)` to the Pareto-optimal (loss envelope) points. Running every combination is expensive. This repo instead:

1. Trains a **surrogate model** on a small sampled subset of runs (in log-space over `N`, `D`, `lr`, `gbsz`).
2. Uses the surrogate to **predict loss** for all configurations cheaply.
3. Extracts the **Pareto frontier** (best loss per compute budget `C = 6ND`) from the predictions.
4. Fits a **power law** to the envelope and reports `A` and `alpha`.

## Files

| File | Description                                                                                                     |
|---|-----------------------------------------------------------------------------------------------------------------|
| `model_based_fits.py` | Main entry point — runs the surrogate pipeline and writes results to JSON                                       |
| `models.py` | Surrogate model backends: GP (BoTorch/GPyTorch), TabPFN, AutoGluon                                              |
| `selection.py` | Training data selection strategies: random, local search                                                        |
| `fit_scaling_laws_loss_vs_compute.py` | Standalone script using the original code from Niccolo for fitting and plotting a power law on the full dataset |
| `preprocessing/fetch_data.py` | Downloads the raw CSV from each source repo into `data/raw/` |
| `preprocessing/preprocess.py` | Adapts each raw CSV into the common `N, D, lr, gbsz, loss` schema in `data/preprocessed/` |

## Usage

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
| `steplaw` | Li et al., arXiv:2503.04715 | yes |
| `porian` | Porian et al., arXiv:2406.19146 | yes |
| `gemstones` | McLeish et al., arXiv:2502.06857 | no |
| `activesl_lr_bsz` | active-experiment-selection benchmark, arXiv:2604.22753 | yes |
| `activesl_farseer` | active-experiment-selection benchmark, arXiv:2604.22753 | no |
| `activesl_data_constrained` | active-experiment-selection benchmark, arXiv:2604.22753 | no |
| `warmstart` | Mallik et al., arXiv:2605.13405 | yes |

Sources with no `lr`/`gbsz` in the underlying data leave those columns blank -- usable with `fit_scaling_laws_loss_vs_compute.py` but not `model_based_fits.py`/`iterative_fits_L0_form.py`, which need all four features.

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
| `N_total` | Every trainable parameter (embedding/head counted once if tied, twice if untied). Porian et al. (arXiv:2406.19146) call this `N_exact` -- the name the `porian_*.csv` files use for their own column. |
| `N_ne` | Excludes both the input embedding and output head/unembedding matrices ("non-embedding"). Porian et al. call this `N_Kaplan`, after Kaplan et al.'s original convention -- the name the `porian_*.csv` files use for their own column. |
| `N_Porian` | Porian et al.'s own default: excludes the input embedding but includes the head; close to `N_total` when embeddings are tied, since the head then reuses the embedding matrix. Only appears in `porian_*.csv`. |
| `N_eff` | Porian et al.'s "effective" parameter count: `N_Porian` adjusted upward for attention-operation FLOPs (bigger relative effect on smaller models). Only appears in `porian_*.csv`. |

### Surrogate-based fit

```bash
python model_based_fits.py \
    --input      path/to/all_points_clean.csv \
    --output_dir results/ \
    --num_points 50 \          # absolute number of training runs
    --seed       1 \
    --model_type gp \          # gp | tabpfn | ag
    --selection  random        # random | local_search
```

Results are written to `output_dir` as `results_{model_type}_{selection}_{num_points}_points_seed_{seed}.json`. Each file contains the fitted `A_fit`, `alpha_fit`, surrogate MSE, number of training points, and total compute of the training subset.

### Full-data fit and plot

```bash
python fit_scaling_laws_loss_vs_compute.py
```

Reads `data/preprocessed/oellm_english.csv`, fits a power law on the full loss envelope, and saves a plot to `results/fit_compute.png`.

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
| `gp` | `GPModel` | BoTorch + GPyTorch (Matérn kernel, ARD) |
| `tabpfn` | `TabPFNModel` | TabPFN |
| `ag` | `AutoGluonModel` | AutoGluon TabularPredictor |

All models receive log-transformed features.

## Selection strategies

| Key | Class | Description |
|---|---|---|
| `random` | `RandomSelection` | Uniformly samples `num_points` runs at random |
| `local_search` | `LocalSearch` | Starts at the smallest scale and performs coordinate-wise local search over `lr`/`gbsz`, transferring the incumbent to larger `N`/`D` scales |

## Dependencies

Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

Per surrogate backend:
- GP: `botorch`, `gpytorch`, `torch`
- TabPFN: `tabpfn`
- AutoGluon: `autogluon.tabular`
