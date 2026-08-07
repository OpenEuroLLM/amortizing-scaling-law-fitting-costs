# Step Law dataset

Li et al., "Predictable Scale: Part I, Step Law -- Optimal Hyperparameter
Scaling Law in Large Language Model Pretraining"
([arXiv:2503.04715](https://arxiv.org/abs/2503.04715)), data and code at
https://github.com/step-law/steplaw.

Li et al. train dense and MoE models from scratch across ~100T tokens to
fit a joint (lr, gbsz) scaling law (Sec. 1). The collected data
span 215M-2.16B parameters and 2B-100B token budgets,
with an (lr, gbsz) grid search per (N, D) pair at a fixed 2,048-token
sequence length and an LR schedule of linear warmup over the first 2,000
steps then cosine decay to a fixed floor of 1e-5 (Sec. 3.2, 3.3.2).

## The data

| upstream file | our dataset name | points | configs covered |
|---|---|---|---|
| `dense_lr_bs_loss.csv` | `steplaw_dense_n1911` | 1,911 | 17 of Table 5's 18 dense configs: 5 architecture shapes (960-2048 hidden dim, 7-16 layers), 215M-1.07B non-embedding parameters, token budgets from 4B to 100B, 47-120 (lr, gbsz) combinations per config -- not a fixed factorial grid |
| `moe_lr_bs_loss.csv` | `steplaw_moe_n708` | 708 | 16 of Table 6's 18 MoE configs: 4 sparsity configs (89/88/8/8 experts) at a fixed ~2.15B total parameters, token budgets from 2B to 20B |

## Columns

Besides the common `N, D, lr, gbsz, loss`, both adapters (with `--extended`)
rename and keep `width`, `depth`, `ffn`, `num_heads` from the raw files'
`h`, `numl`, `ffnh`, `numh`:

- `N` = the paper's non-embedding parameter count (Appendix A.1); for the
  dense file this checks out almost exactly against a standard attention
  (`4*d_model^2`) + SwiGLU-MLP (`3*d_model*d_ff`) parameter count, i.e. it
  excludes the embedding and head entirely -- recorded here as
  `N_definition = "N_ne"` (see README.md's "N-variant vocabulary" section).
- `D` = dataset size in tokens (Appendix A.1); matches
  `gbsz * 2048 * ti` (batch size x sequence length x training steps) to
  within ~0.2%, so `ti` is the step count needed to reach that `D`, rounded.
- `gbsz` = `bs` renamed; global batch size in sequences at the fixed
  2,048-token sequence length -- the paper's own `BS` (batch size in
  tokens, Appendix A.1) is `gbsz * 2048`.
- `loss` = the run's final logged-step loss, not the `smooth loss` column
  also present in the raw files (the two correlate at ~0.9997). Sec. 3.3.3
  argues smooth training loss is the better proxy for validation loss,
  following Chinchilla's precedent -- so this differs from what the paper
  treats as its primary metric.
- `width`/`depth`/`ffn`/`num_heads` = `h`/`numl`/`ffnh`/`numh` =
  `d_model`/`N_layer`/`d_ff`/`N_head`.
- `seq_len` = 2,048 (Sec. 3.2); a constant for `steplaw_dense_n1911`, read
  directly from the `seq_len` column for `steplaw_moe_n708`.
- `warmup_iters` = 2,000, `lr_schedule` = `"cosine decay"` (Sec. 3.2); both
  paper-stated constants, not present as columns in either file.

`steplaw_moe_n708` additionally renames and keeps `nume`/`moeh`/`topk`/`Na`:

- `num_experts`/`expert_width`/`top_k` = the number of routed experts, each
  expert's hidden dimension, and the routing top-k (Table 6).
- `N_active` = active parameters per token (Table 6's `N_a`).
