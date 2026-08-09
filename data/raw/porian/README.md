# Porian dataset

Porian et al., "Resolving Discrepancies in Compute-Optimal Scaling of
Language Models" ([arXiv:2406.19146](https://arxiv.org/abs/2406.19146)),
repository at
https://github.com/formll/resolving-scaling-law-discrepancies.

Porian et al. provide a collection of 975 training runs across 16 fixed
(depth, width) architectures (5M-900M parameters), pretrained on either
RefinedWeb or OpenWebText2. All runs share a fixed AdamW optimizer, bf16
precision, and qk-norm setup, with 2,048-token sequences and a 50,432-token
vocabulary.

The raw file's `hparams` column splits the runs into 4 experiments -- the
main compute-optimal grid (`sweep`), a decay-schedule ablation (`base`), a
repeated-seed noise estimate (`seed`), and a tuned-hyperparameter check
(`tuned`) -- and its `dataset` column splits most of those across the 2
pretraining corpora. `preprocess.py` adapts these into one output dataset
per (experiment, corpus, decay schedule) combination that's actually
populated.

## The data

| our dataset name | points | what it covers |
|---|---|---|
| `porian_main_stable_rw_n566` | 566 | The main compute-optimal grid: warmup then a stable (non-decaying) LR, 7 (lr, gbsz) combinations per size. |
| `porian_chinchilla_cosine_decay_rw_n100` | 100 | Baseline runs retrained under a `chinchilla`-style cosine decay (to 1% of peak) instead of a stable LR -- a decay-schedule ablation. |
| `porian_chinchilla_cosine_decay_owt2_n100` | 100 | Same ablation as `porian_chinchilla_cosine_decay_rw_n100`, on OpenWebText2. |
| `porian_kaplan_cosine_decay_rw_n32` | 32 | Baseline runs retrained under a `kaplan`-style cosine decay (to 0) instead of a stable LR -- a decay-schedule ablation. |
| `porian_kaplan_cosine_decay_owt2_n32` | 32 | Same ablation as `porian_kaplan_cosine_decay_rw_n32`, on OpenWebText2. |
| `porian_seed_stable_rw_n42` | 42 | Repeated-seed runs (seeds 1-7, vs. seed 0 elsewhere) at a handful of sizes -- for estimating run-to-run noise. |
| `porian_seed_stable_owt2_n42` | 42 | Same as `porian_seed_stable_rw_n42`, on OpenWebText2. |
| `porian_tuned_stable_rw_n32` | 32 | Runs with per-size tuned (lr, gbsz) -- optimal-hyperparameter sensitivity check. |
| `porian_tuned_stable_owt2_n29` | 29 | Same as `porian_tuned_stable_rw_n32`, on OpenWebText2. |

### Naming convention

```
porian_{purpose}_{schedule}_{dataset}_n{rows}
```
- `{purpose}` = what the runs are for: `main` (the compute-optimal grid,
  raw `hparams=sweep`), `chinchilla`/`kaplan` (the decay-schedule ablation,
  raw `hparams=base` split further by raw `decay`), `seed` (repeated-seed
  noise estimate, raw `hparams=seed`), `tuned` (per-size tuned
  hyperparameters, raw `hparams=tuned`).
- `{schedule}` = `stable` (raw `decay=const`) or `cosine_decay` (raw
  `decay=chinchilla`/`kaplan`).
- `{dataset}` = the raw `dataset` value the variant filters on (`rw`/`owt2`).
- `n{rows}` = row count of the resulting CSV.

## Columns

Besides the common `N, D, lr, gbsz, loss`, `--extended` adds:

- `N`, `N_Porian`, `N_exact`, `N_eff`, `N_Kaplan` = looked up per (depth, width)
  from Table 2, Appendix B of the paper -- see README.md's "N-variant
  vocabulary" section for what each variant means. `N` defaults to
  `N_Porian`.
- `D` = `seq_len * gbsz * max_step` (all three read directly from the raw file).
- `loss` = the last logged `val/loss` value for the run.
- `gbsz` = the raw `bs` column, renamed.
- `width`/`depth` = read directly from the raw file; together they key the
  `N`-variant lookup table.
- `seq_len` = read directly from the raw file; 2,048 for every run.
- `tied_embeddings` = `True` for every run. Not a raw column: confirmed by
  reconstructing `N_exact` as `N_Kaplan` (non-embedding params) plus one
  `vocab_size * width` embedding matrix, which matches the lookup table to
  within ~0.1% across all 16 architectures -- two matrices (untied) would
  overshoot `N_exact` by 8-94%.
- `lr_schedule` = what the LR does after warmup, translated from the raw
  `decay` column: `"stable"` (raw `const`) or `"cosine_decay"` (raw
  `chinchilla`/`kaplan`). Verified against the raw per-step `train/lr`
  curves, which log the LR for the entire run for `chinchilla`/`kaplan` but
  only an early prefix for `const`.
- `decay_iters` = the decay's length in steps, `NaN` for `"stable"` rows.
  Derived, not read directly:
  - `chinchilla` (Sec. 3.4) decays from the end of warmup to its target
    *at that run's own `max_step`* -- confirmed across the whole
    `chinchilla` set (see `min_lr` below). So
    `decay_iters = max_step - warmup_iters`.
  - `kaplan` (Sec. 3.1) decays over a fixed token span
    (`2.5e5 * 2**19` =~ 131B tokens) shared by every `kaplan` run
    regardless of model size, not tied to that run's own `max_step` -- see
    `min_lr` below for how this was confirmed. So
    `decay_iters = ceil((131B_tokens - warmup_tokens) / (seq_len * gbsz))`.
- `min_lr` = the LR floor the decay reaches, `NaN` for `"stable"` rows.
  Derived, not read directly:
  - `chinchilla` targets 1% of peak `lr` (Sec. 3.4) -- confirmed by
    checking that floor/peak in the logged curve converges to ~0.01 right
    at `max_step`, regardless of `max_step`'s value. So
    `min_lr = 0.01 * lr`.
  - `kaplan` targets 0 (Sec. 3.1) -- confirmed by reproducing each run's
    logged floor value from a cosine formula ending at 0 over the fixed
    131B-token span above (matches to 6 decimal places across all 64
    rows). So `min_lr = 0`.
- `warmup_iters` = the raw `warmup_tokens` column converted to steps via
  `warmup_tokens / (seq_len * gbsz)`, rounded up.
- `seed` = training seed (0 outside `porian_seed_stable_{rw,owt2}_n42`, 1-7
  within it).
