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

## Usage

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

Reads `../../data/preprocessed/all_points_clean.csv`, fits a power law on the full loss envelope, and saves a plot to `../../fig/model/fit_compute.png`.

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
