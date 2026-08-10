# Warmstart dataset

Mallik, Janowski, Hog et al., "When is Warmstarting Effective for Scaling
Language Models?" ([arXiv:2605.13405](https://arxiv.org/abs/2605.13405)),
repository at https://github.com/Neeratyoy/warmstarting_exps.

Mallik et al. provide a collection of 170 decoder-only language-model
training runs (14M-1.2B parameters, 8 layers, fixed 64-dim attention heads --
Table 5) comparing three ways of initializing a *target* model of a given
size: training it from scratch, or growing it in width from a smaller,
already-trained *base* checkpoint via one of two warmstarting operators. All
runs are trained on the SlimPajama corpus (Soboleva et al., 2023) using the
open-source LitGPT implementation of a GPT-2-style architecture, for a fixed
budget of 10, 20, or 30 tokens/parameter (Sec. 4.1.2).

All runs share AdamW ((&beta;<sub>1</sub>, &beta;<sub>2</sub>) = (0.9, 0.95),
&epsilon; = 1e-8), zero weight decay, a 1,024-token sequence length, and a
warmup-stable-decay LR schedule with a 1%-of-training linear warmup and a
20%-of-training linear cooldown to 0 (Appendix D.2). Hyperparameters (peak
LR, effective batch size) are tuned at each of 3 base scales and transferred
to target scales via &mu;P (Appendix D.3, Table 6).

The raw file's `method` column bundles the 3 initialization techniques being
compared against the same (N, D) grid, each with its own loss(N, D) surface
by design; `preprocess.py` adapts them into one output dataset per method.

Not hosted in the paper's own repository (linked above); the raw file is
fetched from a Google Drive link the authors share in a sibling project's
README (`EffingScaling/README.md`, "Warmstarting Data").

## The data

| our dataset name | points | method |
|---|---|---|
| `warmstart_scratch_n33` | 33 | Training the target model from a random initialization, with &mu;P-transferred hyperparameters (raw `method="mup"`) -- the paper's `Scratch` baseline. |
| `warmstart_net2net_n27` | 27 | Growing the target model from the base checkpoint via Net2Net (Chen et al., 2016; Appendix G) -- a function-preserving width-expansion baseline. |
| `warmstart_szp_n110` | 110 | Growing the target model from the base checkpoint via Shrink-Zeros-Perturb (SZP, Sec. 3.2, Eq. 2) -- the paper's own architecture-agnostic warmstarting method, and its main object of study. Raw `warmstart_type="snp_zeros_with_mup"`. |

### Naming convention

```
warmstart_{method}_n{rows}
```
- `{method}` = which of the 3 initialization techniques above, from the
  raw `method` column.
- `n{rows}` = row count of the resulting CSV.

## Columns

Besides the common `N, D, lr, gbsz, loss`, `--extended` adds:

- `N` = the raw `target_N` (the post-growth size actually trained and
  evaluated; equal to the from-scratch model size for `warmstart_scratch`).
  Confirmed as each architecture's exact total parameter count: reconstructing
  it from `width`/`depth` as `12*depth*width**2 + 101482*width` (attention
  `4*width**2` plus a 4x-ratio MLP `8*width**2`, no bias terms) matches all 6
  architectures in Table 5 to the exact integer. The paper doesn't state a
  vocabulary size or whether embeddings are tied, so the `101482*width`
  remainder (embedding/head/positional parameters) can't be split further,
  and `N_definition` is recorded as `"N_total"` rather than one of the more
  specific variants.
- `D` = the raw `tokens` column at the run's last checkpoint with a logged
  `Validation Loss` (a couple of runs are missing their true final eval, so
  this collapse isn't always literally the run's last step -- confirmed
  against the nominal `tkpm * target_N` budget, which it matches to within
  0.03% for all but one early-stopped run).
- `loss` = the raw `Validation Loss` at that same checkpoint.
- `lr` = the raw `max_lr` (peak LR after warmup).
- `gbsz` = `micro_batch_size * accumulation_iters * devices`.
- `width`/`depth` = the raw `n_embd`/`n_layer`, renamed.
- `seq_len` = the raw `block_size`, renamed; 1,024 for every run (Table 5).
- `tied_embeddings` = left unset (see `N` above).
- `N_definition` = `"N_total"` for every run (see `N` above).
- `lr_schedule` = `"linear_decay"` for every run -- the warmup-stable-decay
  schedule's cooldown phase is linear (Appendix D.2's
  &eta;<sub>t</sub> = &eta;<sub>max</sub>(T-t)/(dT)).
- `warmup_iters`/`decay_iters` = the raw `warmup_fraction`/`cooldown_fraction`
  (fixed at 0.01/0.2 for every run) times the run's own total step count
  (`D / (seq_len * gbsz)`), rounded to the nearest integer.
- `min_lr` = the raw `min_lr` column; 0 for every run, matching the
  decay-to-zero cooldown above.
- `base_N` = the raw `base_N` -- the pre-growth checkpoint's parameter count
  the run warmstarted from (`warmstart_scratch`'s runs don't inherit any
  weights, but `base_N` still identifies which base scale's hyperparameters
  were &mu;P-transferred to train them, so distinct runs can share `(N, D)`
  and differ only by `base_N`).

## Limitations

- `tied_embeddings` is unset (see `N` above) rather than confirmed one way
  or the other.
