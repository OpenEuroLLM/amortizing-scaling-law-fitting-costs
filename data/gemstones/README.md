# Gemstones dataset

McLeish et al., "Gemstones: A Model Suite for Multi-Faceted Scaling
Laws" ([arXiv:2502.06857](https://arxiv.org/abs/2502.06857)), data and code at
https://github.com/mcleish7/gemstone-scaling-laws.

McLeish et al. train 22 model shapes -- spanning 11 widths (256-3072) and 18
depths (3-80), not a full width x depth grid -- ranging 50M-2B parameters, on
350B tokens of Dolma (Soldaini et al., 2024), using a WSD (warmup-stable-decay)
learning rate schedule, at a fixed batch size of 2048 sequences x 2048-token
context (~4M tokens/batch), checkpointing every 2B tokens for 4,000+
checkpoints total (Sec. 3); the longest (350B-token) runs have up to 175
extracted token budgets.

All 15 dataset variants the paper's repo provides are fetched here (via
`preprocessing/fetch_data.py`) and each
adapted separately into its own `data/gemstones/preprocessed/gemstone_*.csv`
(via `preprocessing/preprocess.py`). 

## The data

Every adapted upstream file shares the prefix `wandb_df_for_fitting_` and
every dataset name we give it shares the prefix `gemstone_`; both are
dropped below and shown as suffixes only.

| upstream file suffix | our dataset name suffix | n_points | model size range (n sizes) | token budget range (n budgets) | what differentiates it |
|---|---|---|---|---|---|
| `hot.jsonl` | `stable_100b` | 220 | 48.4M-2.01B (22 sizes) | 10B-100B (10 budgets) | pretrain, checkpointed every 2385 steps |
| `hot_477.jsonl` | `stable_100b_dense` | 1,100 | 48.4M-2.01B (22 sizes) | 2.00B-100B (50 budgets) | pretrain, checkpointed every 477 steps (5x denser than `stable_100b`) |
| `hot_100b+.jsonl` | `stable_350b` | 770 | 48.4M-2.01B (22 sizes) | 10B-350.1B (35 budgets) | pretrain, checkpointed every 2385 steps |
| `hot_100b+_477.jsonl` | `stable_350b_dense` | 3,850 | 48.4M-2.01B (22 sizes) | 2.00B-350.1B (175 budgets) | pretrain, checkpointed every 477 steps (5x denser than `stable_350b`) |
| `hot_120b+_only.jsonl` | `stable_120to350b` | 528 | 48.4M-2.01B (22 sizes) | 120B-350.1B (24 budgets) | pretrain, only the 120B-350B tail, checkpointed every 2385 steps |
| `hot_120b+_only_477.jsonl` | `stable_120to350b_dense` | 2,552 | 48.4M-2.01B (22 sizes) | 120B-350.1B (116 budgets) | pretrain, only the 120B-350B tail, checkpointed every 477 steps (5x denser) |
| `hot_100b+_512x.jsonl` | `stable_350b_512width` | 175 | 94.8M-114.4M (5 sizes) | 10B-350.1B (35 budgets) | pretrain, filtered to the width=512 model-shape family only |
| `cool_begin.jsonl` | `stable_100b_predecay` | 199 | 48.4M-2.01B (20 sizes) | 10B-100B (10 budgets) | decay runs, checkpoint at the pre-branch point (same steps as `stable_100b`) |
| `cool_end.jsonl` | `decay_11to110b` | 199 | 48.4M-2.01B (20 sizes) | 11B-110B (10 budgets) | decay runs, checkpoint at decay end (steps x1.1 past the branch) |
| `lr_ablation_hot.jsonl` | `stable_100b_lrablation` | 200 | 48.4M-2.01B (20 sizes) | 10B-100B (10 budgets) | LR-ablation (half-LR sweep), checkpointed every 2385 steps |
| `lr_ablation_hot_477.jsonl` | `stable_100b_dense_lrablation` | 1,000 | 48.4M-2.01B (20 sizes) | 2.00B-100B (50 budgets) | LR-ablation, checkpointed every 477 steps (5x denser) |
| `hot_sampled_like_chinchilla.jsonl` | `stable_sampled_like_chinchilla` | 24 | 66.8M-1.91B (5 sizes) | 2.00B-124B (13 budgets) | pretrain, downsampled to match Chinchilla's own (N, D) density |
| `lr_ablation_hot_sampled_like_chinchilla.jsonl` | `stable_lrablation_sampled_like_chinchilla` | 23 | 66.8M-1.91B (5 sizes) | 2.00B-82.0B (12 budgets) | LR-ablation, downsampled to match Chinchilla's own (N, D) density |
| `slim_chinchilla.jsonl` | `external_chinchilla` | 28 | 73.8M-2.01B (15 sizes) | 0.81B-124.5B (28 budgets) | **not Gemstones data** -- Epoch AI's Chinchilla-replication points, comparison baseline |
| `lr_ablation_slim_chinchilla.jsonl` | `external_chinchilla_lrablation` | 27 | 73.8M-2.01B (15 sizes) | 0.81B-81.3B (27 budgets) | **not Gemstones data** -- same, baseline for the LR-ablation runs |

### Naming convention

```
gemstone_{stable|decay|external}[_{token range}][_dense][_axis]
```

- `stable` = constant LR schedule throughout (no decay).
- `decay` = loss measured after an LR-annealing decay phase (the paper's
  `cooldown`-tagged branch runs).
- `external` = not a Gemstones run at all -- Epoch AI's Chinchilla-replication
  data, redistributed by the paper's repo as a comparison baseline.
- `[_{token range}]` = the span of tokens the checkpoints cover, e.g. `100b`,
  `350b`, `120to350b`, `11to110b`; omitted for the Chinchilla-sampled
  variants, which don't follow a fixed token ladder.
- `[_dense]` = marks the 5x-denser checkpoint variant (every 477 steps
  instead of every 2385) of a token range that also has a base variant.
- `[_axis]` = a trailing qualifier when a file isn't just "the base ladder":
  `lrablation` (half-LR sweep), `512width` (filtered to the width=512
  model-shape family), `predecay` (decay runs, pre-branch checkpoint),
  `sampled_like_chinchilla` (downsampled to match Chinchilla's own (N, D)
  density). Multiple qualifiers stack in this order, e.g.
  `..._dense_lrablation` or `..._lrablation_sampled_like_chinchilla`.

## Columns

Besides the common `N, D, loss`, `adapt_gemstones` fills in `lr`, `gbsz`, `width`,
`depth`, `seq_len`, `warmup_iters`, `decay_iters`, `tied_embeddings`,
`N_definition` from constants and formulas given in the paper:

- `gbsz` = 2048 sequences and `seq_len` = 2048 tokens, both fixed for the
  entire sweep (Sec. 3).
- `lr` = `lr_base / (width * sqrt(depth))` (Appendix A.1's hybrid muP-style
  rule for transferring a learning rate across both width and depth at
  once), constant after an 80M-token warmup. `lr_base = 5` for the main
  (`stable`/`decay`) runs, empirically found to be the shared optimum across
  shapes; `lr_base = 2.5` for the `lrablation` variants, which the paper
  runs specifically at half of `lr_eff`. For `decay` rows this is the
  pre-decay reference rate, not the instantaneous rate (the LR is annealed
  to 0 over the decay phase).
- `warmup_iters` = `ceil(80_000_000 / (seq_len * gbsz))` = `20` steps (Sec. 3).
- `decay_iters`: `predecay`/`decay` checkpoints are branches off 20 of the 22
  shapes' `stable_100b` pretrain runs (not a separate run) -- at 10
  points along that run's schedule (every 2385 steps, up to ~100B tokens), a
  branch continues training for an extra ~10% of steps under a decay
  schedule. `0` for `predecay` rows (the branch point), `step - step / 1.1`
  for `decay` rows (after the branch); `None` for every other variant.
- `tied_embeddings` = `False` for every Gemstones-native row (Appendix A).
- `N_definition` = `"N_total"` (see README.md's "N-variant vocabulary" section).
