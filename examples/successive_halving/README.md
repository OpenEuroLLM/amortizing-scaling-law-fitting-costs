# Successive Halving baseline

Strategic compute allocation for scaling-law estimation, after **"Active Budget
Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning"**
(Schram et al., arXiv:2605.17234). Given a fixed compute budget `B` and a set of
model sizes, decide *how much compute to spend on each model* so that the
resulting learning curves recover an accurate loss–compute frontier.

Three strategies (paper Section 3):

| Strategy | How it spends `B` |
|---|---|
| **Uniform** | Split `B` equally across all models; train each to `B/M`, no pruning. |
| **Successive Halving (SH)** | Rounds of train-a-bit / keep-the-best-half. Rank survivors by their **current observed loss**. |
| **SH + surrogate** | Same rounds, but rank by a surrogate's **predicted final loss** — so a promising model that merely looks bad *early* isn't pruned. |

## Why synthetic data

SH needs densely-sampled learning curves (loss vs compute *along* a model's
training). The repo's `steplaw` grid only has 1–3 duration points per config, so
we generate the paper's own testbed instead: learning curves from a known
`L(N,D)=E+A·N^-α+B·D^-β` law (Hoffmann/Chinchilla coefficients), with
`C = 6ND`. Because the law is known, we have a gold-standard frontier to score
each strategy's recovered scaling law against.

## Contents

The allocator itself lives in the repo's canonical `selection.py`, as the
`SuccessiveHalving` class (alongside `RandomSelection` / `LocalSearch`). It reads
learning curves from an `oracle(N, C) → loss`, so it stays decoupled from *how*
the loss is produced; this example passes the synthetic `L(N,D)` oracle.

| File | Role |
|---|---|
| `../../selection.py` | `SuccessiveHalving` — the `uniform` / SH / SH+surrogate allocators. |
| `synth_lc.py` | The `L(N,D)` law + oracle, per-model learning curves, and the true compute-optimal frontier. |
| `surrogate.py` | GP over `(log N, log C)→log loss` — the SH surrogate (stand-in for the paper's LMC/Deep-Ensemble). |
| `fit_eval.py` | Fit `L(C)=E+A·C^b` on a strategy's frontier; score `l*` and frontier error vs. the true law. |
| `run.py` | Sweep seeds × strategies; report relative improvement, as in the paper's tables. |
| `plot.py` | Figure-1 replica: the curves each strategy trained, its kept models, and `l*`. |
| `plot_density.py` | Motivation figure: synthetic (~20 D/model) vs steplaw (2–4 D/model), with fitted `L(C)`. |
| `acquire.py` | **SH-as-GP-acquisition vs. random** on any grid (`--grid`, `--law`) (see below). |
| `make_synth_grid.py` | Write a dense synthetic grid CSV (identifiable exponents) for `acquire.py`. |

## SH as a GP-guided acquisition step (real grid)

The synthetic package above runs *batch* Successive Halving on the paper's dense
testbed. `selection.py: SuccessiveHalvingGP` is a different, complementary
integration: SH's surrogate-guided pruning expressed as a **sequential
acquisition step** that reads real `(N, D, lr, gbsz)` grid rows — so it drops
into the same iterative loop as the other baselines and pays for hyperparameters
like they do (the GP picks the config; no free oracle HP).

Each `next_point`: consider only unopened `(N, D)` rungs → score each model size
by the GP's best achievable predicted loss (its *potential*) → keep the best size
→ advance it to its next-larger `D`, letting the GP pick the `(lr, gbsz)` there.

```bash
uv run python examples/successive_halving/acquire.py --seeds 0-9
```

`acquire.py` runs `sh_gp` and `random` from a shared seed (one cheapest run per
size), and after each pick fits a scaling law on the observed compute-optimal
points, scoring how well it recovers the **full-grid** law (the ground truth)
over the compute explored so far. It reports every coefficient's trajectory
against its ground-truth value. Figure + numbers land in
`results/baselines/successive_halving/acquire_steplaw/`.

`--law` selects the form: `lc_l0` = `E+A·C^b` (3-param, most stable here),
`lc_power` = `A·C^b` (2-param), `nd` = `E+A·N^-α+B·D^-β` (two exponents). The
`nd` form exposes a real limit of this grid: with only 5 model sizes over ~5× in
`N`, the full-grid fit itself cannot identify the model exponent `α` or the floor
`E` (both degenerate); only the data exponent `β` is recoverable. Fewer, denser
sizes make `L(C)` far better behaved than `L(N,D)` here.

### On a dense synthetic grid (`--grid`)

`make_synth_grid.py` writes a dense grid (16 model sizes over a wide `N` range, 8
durations each, 3 HP configs/cell) from the same `L(N,D)` oracle, so the
exponents *are* identifiable. Point `acquire.py` at it:

```bash
uv run python examples/successive_halving/make_synth_grid.py
uv run python examples/successive_halving/acquire.py \
    --grid results/baselines/successive_halving/data/synth_grid.csv --law nd --seeds 0-9 --niters 40
```

Here the full-grid fit recovers the generating law (`α≈0.339, β≈0.273, E≈1.66`)
and **both** methods converge to it (recovery ~1e-4, 100× better than steplaw).
Notably, **random is competitive-to-better on coefficient recovery**: identifying
`L(N,D)` rewards *diverse* model-size coverage, whereas `SuccessiveHalvingGP`
deliberately *concentrates* compute on the most-promising size. That is the
point — SH optimizes finding the lowest-loss model (`l*`) under a scarce budget
(where its concentration wins, see the steplaw frontier-recovery result and the
synthetic `run.py` `l*` result), **not** identifying the full scaling law, which
favors spread. The two testbeds together show both sides honestly.

**Finding (steplaw, held-out largest N):** SH-GP recovers the frontier with *less
compute* than random (blue below gray in the recovery panel) — the paper's claim,
visible even here. But the fitted **exponent** is too noisy to separate the two:
with only 5 model sizes and a <2-decade compute range, the 3-parameter fit is
underdetermined early (SH-GP's `b` swings to −0.48 before settling near the
−0.183 full-grid reference). Treat this as an honest illustration on real data,
**not** evidence about the methods — the grid is too sparse for a clean verdict.

Caveats specific to this run: it's a *sequential* adaptation of SH (advance the
most-promising curve one rung at a time), not literal batch rungs; the recovery
metric is evaluated over each run's own explored compute range, so equal-compute
comparisons are approximate.

## Quick start

```bash
uv run python examples/successive_halving/run.py          # 16 models, B=1e20, σ=0.3, seeds 0–19
uv run python examples/successive_halving/plot.py --seed 8 --noise 0.4
```

Useful overrides: `--n-models`, `--budget`, `--eta` (pruning factor), `--noise`
(std of observation noise), `--seeds 0-49`. Results and figures land under
`results/baselines/successive_halving/`.

## What to expect

- **SH beats uniform** on the loss reached (`l*`): ~8% mean improvement here.
  Uniform trades a lower `l*` for denser frontier coverage.
- **The surrogate's edge is in noise.** With clean curves the compute-optimal
  model barely moves, so greedy SH is already near-optimal and the surrogate
  ties it. Add observation noise and greedy SH prunes good models by chance,
  while the surrogate — fit across all revealed points — smooths over it:
  ~1% mean / ~5% max improvement in `l*` at σ=0.5, and up to ~8× lower frontier
  error. This matches the paper's headline (mean/max up to 2.84%/5.47%).

## Simplifications vs. the paper

- **Surrogate.** We use a single joint GP over `(log N, log C)`; the paper uses a
  multitask GP (LMC) and Deep Ensembles with PL/EXP/MMF curve forms. The joint
  GP captures the same cross-curve correlation without assuming the generating
  form.
- **Noise.** One i.i.d. Gaussian level on the observed loss; the paper also
  studies Ornstein–Uhlenbeck and Brownian noise.
- **Real data.** Synthetic `L(N,D)` only; the paper also runs real learning-curve datasets.
