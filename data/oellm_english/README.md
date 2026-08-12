# OELLM English dataset

Niccolò Ajroldi, "Scaling of OpenEuroLLM Dense English Models" (technical
report, `OELLM___HP_Scaling_Laws.pdf` in the repository root), repository at
https://github.com/OpenEuroLLM/dense_english_scaling_results.

Ajroldi provides a collection of dense decoder-only architectures at 5 sizes
(47.6M-983.1M non-embedding parameters, Table 1) plus a 6th, 1.7B-parameter
size held out as an extrapolation test point (not in Table 1), pretrained on
the high-quality subset of Nemotron-CC up to 300BT with intermediate
checkpoints decayed at 9 token budgets (D = 6, 12, 20, 30, 50, 80, 120, 200,
300 BT), over a learning-rate x batch-size grid per (N, D) pair, for 890
training runs total (Sec. 1.1).

All runs share AdamW, a 50,000-token vocabulary, FFN expansion factor 4 with
GLU activations, attention bias terms, QK normalization, tied input/output
embeddings, a 4,096-token sequence length, and a Warmup-Stable-Decay (WSD,
"trapezoidal") learning-rate schedule: 2,000-step fixed warmup, then a
linear decay covering the final 20% of each run's own steps down to a floor
of 1e-5 (Sec. 1.1, Sec. 3).

## Columns

Besides the common `N, D, lr, gbsz, loss` (read directly from the raw file),
`--extended` adds:

- `N` = the paper's non-embedding parameter count (`data/preprocessed/README.MD`
  in the source repository) -- recorded as `N_definition = "N_ne"` (see
  README.md's "N-variant vocabulary" section). Matches Table 1's "Total
  parameters" row exactly, and is independently reconstructed as
  (total parameters including the tied embedding) minus
  `vocab_size(50,000) * width`, exactly, for all 5 architectures (the
  source's own `scripts/preprocess.ipynb`).
- `width`/`depth`/`num_heads`/`ffn` = Table 1's "Model dimension"/"Layers"/
  "Attention heads"/"FFN dimension", looked up per exact `N`. `None` for the
  held-out 1.7B rows, which aren't in Table 1.
- `tied_embeddings` = `True` for every run (Sec. 1.1).
- `seq_len` = 4,096 (the source's `SEQ_LENGTH` constant); independently
  confirmed by reconstructing the raw `train_tokens` column exactly as
  `iteration * gbsz * 4096` across all 890 rows.
- `vocab_size` = 50,000, fixed across every run (Sec. 1.1).
- `lr_schedule` = `"warmup_stable_decay"` for every run: the WSD decay phase is linear.
- `warmup_iters` = 2,000, fixed across every run (Sec. 1.1).
- `decay_iters` = each run's own final step (`iteration`) minus
  `floor(iteration * 0.8)` -- the decay covers the final 20% of a run's own
  steps (Sec. 3), following the source's own branching-checkpoint formula.
- `min_lr` = 1e-5, fixed across every run (Sec. 3: "always annealing to a
  final learning rate of 1e-5").
- `beta2` = the raw Adam beta2 column (0.95 or 0.99). Not a full ablation
  grid: each model size in this dataset trains at one fixed beta2 (0.99 for
  50M/130M/600M, 0.95 for 300M/1B/1.7B) -- `beta2_analysis.ipynb` in the
  source repository studies its effect separately.
- `seed` = the raw training seed. 1234 for almost every run; 40 runs at
  N=1B, lr=0.001 use seed 2345 instead, deliberately replacing a
  seed-1234 subset the source's own cleaning step (`scripts/clean.ipynb`)
  drops as an outlier ("weird seed, high loss").

## Limitations

- The 1.7B rows are a designated extrapolation test set (Sec. 4-5 fit scaling
  laws on the 5 Table-1 sizes and evaluate on 1.7B), not part of the paper's
  main hyperparameter-scaling grid -- and lack architecture columns, as noted
  above.
- The source repository also has `job_infos.csv`/`results_predecay.csv`/
  `results_stable.csv`, which capture the same runs' pre-decay and
  stable-phase (never decayed) loss instead of the final post-decay loss
  collected here -- not currently fetched or adapted.
