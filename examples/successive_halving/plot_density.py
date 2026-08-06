#!/usr/bin/env python3
"""Why SH wants dense curves: synthetic (~20 D/model) vs the steplaw grid (2-4 D/model).

Same 5 model sizes on both panels, loss vs compute. Left: the synthetic oracle
sampled at 20 points per model (what SH prunes over). Right: the real grid's
compute-optimal loss at each available D -- 2-4 stitched points per model.

Each panel also fits the scaling law L(C)=E+A*C^b to the loss-compute frontier
and overlays it, next to a reference fit (synthetic: the true oracle frontier;
steplaw: the full-grid frontier) so the recovered exponent/coefficients compare.

    uv run python examples/successive_halving/plot_density.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_eval import fit_law
from synth_lc import loss_at_compute, true_frontier

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "baselines" / "successive_halving" / "figures"


def frontier(C, L):
    """Lower envelope: running-min loss over increasing compute (strictly decreasing)."""
    o = np.argsort(C)
    C, m = C[o], np.minimum.accumulate(L[o])
    keep = np.concatenate([[True], np.diff(m) < 0])
    return C[keep], m[keep]


def law(C, p):
    E, A, b = p
    return E + A * C ** b


df = pd.read_csv(REPO / "data" / "steplaw.csv")
df["C"] = 6.0 * df["N"].astype(float) * df["D"].astype(float)
Ns = sorted(df["N"].unique())
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(Ns)))

fig, (aL, aR) = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
fig.suptitle("Learning curves SH prunes over, with fitted L(C)=E+A·C^b",
             fontsize=13, fontweight="bold")

syn_pts, real_pts = [], []
for N, col in zip(Ns, colors):
    sub = df[df.N == N]
    best = sub.loc[sub.groupby("D")["loss"].idxmin()].sort_values("C")
    C_real, L_real = best["C"].to_numpy(), best["loss"].to_numpy()
    C_syn = np.geomspace(C_real.min(), C_real.max(), 20)
    L_syn = loss_at_compute(N, C_syn)
    real_pts.append((C_real, L_real)); syn_pts.append((C_syn, L_syn))

    lab = f"N={N/1e6:.0f}M"
    aL.plot(C_syn, L_syn, "-o", color=col, ms=4, lw=1.3, alpha=0.8, label=f"{lab}  (20 D)")
    aR.plot(C_real, L_real, "-o", color=col, ms=7, lw=1.0, alpha=0.8, label=f"{lab}  ({len(C_real)} D)")

# ---- fits ----------------------------------------------------------------
C_syn_all = np.concatenate([c for c, _ in syn_pts]); L_syn_all = np.concatenate([l for _, l in syn_pts])
C_real_all = np.concatenate([c for c, _ in real_pts]); L_real_all = np.concatenate([l for _, l in real_pts])

p_syn = fit_law(*frontier(C_syn_all, L_syn_all))                       # synthetic observed-frontier fit
Cref = np.geomspace(C_syn_all.min(), C_syn_all.max(), 200)
p_syn_true = fit_law(Cref, true_frontier(Cref))                        # synthetic true-law fit (reference)
p_real = fit_law(*frontier(C_real_all, L_real_all))                    # steplaw full-grid frontier fit

Cg_L = np.geomspace(C_syn_all.min(), C_syn_all.max(), 200)
aL.plot(Cg_L, law(Cg_L, p_syn), "-", color="#dc2626", lw=2.2, zorder=6, label="fit (frontier)")
aL.plot(Cg_L, law(Cg_L, p_syn_true), "--", color="#111827", lw=1.6, zorder=6, label="true-law fit")
Cg_R = np.geomspace(C_real_all.min(), C_real_all.max(), 200)
aR.plot(Cg_R, law(Cg_R, p_real), "-", color="#dc2626", lw=2.2, zorder=6, label="fit (full-grid frontier)")


def box(ax, lines):
    ax.text(0.03, 0.03, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))


box(aL, [f"frontier fit:  E={p_syn[0]:.3f}  A={p_syn[1]:.2e}  b={p_syn[2]:+.4f}",
         f"true law   :  E={p_syn_true[0]:.3f}  A={p_syn_true[1]:.2e}  b={p_syn_true[2]:+.4f}"])
box(aR, [f"full-grid fit: E={p_real[0]:.3f}  A={p_real[1]:.2e}  b={p_real[2]:+.4f}"])

for ax, title in [(aL, "Synthetic oracle  —  ~20 D per model"),
                  (aR, "steplaw grid  —  2–4 D per model")]:
    ax.set_xscale("log")
    ax.set_xlabel("compute C = 6ND (FLOPs)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, title="model size", loc="upper right")
aL.set_ylabel("loss L")

OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "density_synthetic_vs_steplaw.png"
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
print("\nfitted L(C) = E + A * C^b   (b = exponent)")
print(f"  synthetic  frontier fit : E={p_syn[0]:.4f}  A={p_syn[1]:.4e}  b={p_syn[2]:+.4f}")
print(f"  synthetic  true law     : E={p_syn_true[0]:.4f}  A={p_syn_true[1]:.4e}  b={p_syn_true[2]:+.4f}")
print(f"  steplaw    full-grid fit: E={p_real[0]:.4f}  A={p_real[1]:.4e}  b={p_real[2]:+.4f}")
