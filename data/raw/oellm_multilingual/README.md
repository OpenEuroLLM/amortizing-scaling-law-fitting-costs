# OELLM Multilingual dataset

OpenEuroLLM, dense multilingual (Qwen3-architecture) scaling results,
repository at
https://github.com/OpenEuroLLM/dense_multilingual_models_scaling_results.

The repository provides a collection of 329 training runs across 4 dense
Qwen3-architecture sizes (0.21B-1.26B total/`N_Porian` parameters, repo-root
`README.md`), pretrained on a multilingual data mix up to 300BT, with
intermediate checkpoints decayed at D = 6, 12, 20, 30, 50, 80, 120, 200,
300 BT. Each (N, D) pair uses a sparse "1 center + 4 cross" hyperparameter
design (one run at the center (lr, gbsz), plus one-at-a-time lr/gbsz
neighbors) rather than a full grid (`scaling_laws/README.md`). The grid is 
further expanded.

All runs share a 4,096-token sequence length and tied input/output
embeddings (`untie_embeddings_and_output_weights: False`), and a
Warmup-Stable-Decay learning-rate schedule whose linear decay covers the
final 20% of each run's own steps (`scaling_laws/hp_selection/utils.py`).

## Columns

Besides the common `N, D, lr, gbsz, loss` (`N` read from the raw
`N_Porian_exact`, `D`/`lr`/`gbsz`/`loss` read directly), `--extended` adds:

- `N` defaults to `N_Porian_exact` -- the source repository's own
  `N_Porian` convention (repo-root `README.md`, following Porian et al.,
  arXiv:2406.19146): the input embedding is excluded but the output head is
  kept. All runs here use tied embeddings, so `N_Porian` equals `N_total`
  exactly (confirmed against the raw `N_total_exact` column). Recorded as
  `N_definition = "N_Porian"`.
- `N_ne`/`N_total` = the raw `N_ne_exact`/`N_total_exact` columns, renamed to
  match this project's N-variant vocabulary (README.md's "N-variant
  vocabulary" section) -- the source repository's own `README.md` defines
  `N_ne` (transformer-block-only, excluding both embedding matrices) and
  `N_total` (every trainable parameter) identically to that vocabulary.
- `tied_embeddings` = `True` for every run (repo-root `README.md`).
- `seq_len` = 4,096 (`scaling_laws/preprocess_decay.py`'s `SEQ_LEN` /
  `hp_selection/utils.py`'s `compute_checkpoint_iter`).
- `lr_schedule` = `"linear_decay"` for every run: the WSD decay phase is linear.
- `warmup_iters` = 2,000, fixed across every run.
- `decay_iters` = each run's own final step (`iteration`) minus
  `floor(iteration * 0.8)` -- the decay covers the final 20% of a run's own
  steps (`hp_selection/utils.py`'s `compute_checkpoint_iter` docstring,
  "constant across the whole multilingual_scaling model family").
- `min_lr` = 0, fixed across every run.

## Limitations

- No architecture columns (`width`/`depth`/etc.): the repository doesn't
  include a Table-1-style architecture breakdown for the 4 Qwen3 sizes.
- The dataset only has 175 of an expected 180 (N, D, HP) combinations -- the
  5 missing runs are pending evaluation or deliberately not finished
  (`scaling_laws/README.md`).
- The 1.26B size is reserved as a held-out test point in the source's own
  hyperparameter-fitting notebooks (`scaling_laws/README.md`), though it's
  included in this dataset like every other size.
- The source repository also has `training/results/val_loss/predecay.csv`
  and `stable.csv`, which capture the same runs' pre-decay and stable-phase
  (never decayed) loss instead of the final post-decay loss collected
  here -- not currently fetched or adapted.
