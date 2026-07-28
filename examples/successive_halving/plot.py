#!/usr/bin/env python3
"""Figure-1 replica: how uniform / SH / SH-surrogate spend the same budget.

Each panel shows the learning curves a strategy trained (green), the model(s)
it kept for the final round (red), the lowest loss it reached (x), and the
gold-standard compute-optimal frontier of the true L(N, D) (dashed).

    uv run python examples/successive_halving/plot.py --seed 3 --noise 0.3
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))                      # for `selection`, `models`
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for sibling modules
from selection import SuccessiveHalving
from fit_eval import evaluate
from surrogate import gp_surrogate
from synth_lc import loss_at_compute, sample_models, true_frontier

OUT = REPO / "results" / "baselines" / "successive_halving" / "figures"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--n-models", type=int, default=16)
    p.add_argument("--budget", type=float, default=1e20)
    p.add_argument("--eta", type=float, default=2.0)
    p.add_argument("--noise", type=float, default=0.3)
    args = p.parse_args()

    models = sample_models(args.n_models, args.seed)
    sh = SuccessiveHalving(seed=args.seed, eta=args.eta)
    runs = {
        "Uniform allocation": sh.uniform(models, args.budget, loss_at_compute),
        "Successive Halving": sh.run(models, args.budget, loss_at_compute, noise=args.noise),
        "SH + GP surrogate": sh.run(models, args.budget, loss_at_compute,
                                    surrogate=gp_surrogate, noise=args.noise),
    }
    runs = {k: evaluate(v) for k, v in runs.items()}

    Call = np.concatenate([c["Cs"] for r in runs.values() for c in r["curves"]])
    Cref = np.geomspace(Call.min(), Call.max(), 200)
    Lref = true_frontier(Cref)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    fig.suptitle(f"Budget B={args.budget:.0e} FLOPs across {args.n_models} models "
                 f"(seed {args.seed}, noise σ={args.noise})", fontsize=13, fontweight="bold")
    for ax, (name, r) in zip(axes, runs.items()):
        active = set(r["active"] or [])
        for j, c in enumerate(r["curves"]):
            kept = j in active
            ax.plot(c["Cs"], c["Ls"], color="#dc2626" if kept else "#16a34a",
                    lw=2.2 if kept else 1.0, alpha=0.95 if kept else 0.5, zorder=3 if kept else 2)
        ax.plot(Cref, Lref, ls="--", color="#111827", lw=1.4, zorder=4, label="true frontier")
        # min loss reached
        jbest = int(np.argmin([c["Ls"][-1] for c in r["curves"]]))
        cb = r["curves"][jbest]
        ax.scatter([cb["C_end"]], [cb["Ls"][-1]], marker="x", s=90, color="#111827", lw=2.5, zorder=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{name}\nmin loss L*={r['l_star']:.3f}   frontier log-MSE={r['frontier_mse']:.1e}",
                     fontsize=10)
        ax.set_xlabel("compute C (FLOPs)")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("loss L")
    axes[0].legend(loc="lower left", fontsize=9)
    # green/red legend proxy on the last panel
    from matplotlib.lines import Line2D
    axes[-1].legend(handles=[Line2D([], [], color="#16a34a", lw=2, label="trained (pruned earlier)"),
                             Line2D([], [], color="#dc2626", lw=2, label="kept for final round")],
                    loc="lower left", fontsize=9)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"figure1_seed{args.seed}_noise{args.noise}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)
    for name, r in runs.items():
        E, A, b = r["law"]
        print(f"  {name:22s} L*={r['l_star']:.3f}  law: L={E:.3f}+{A:.2e}*C^{b:.3f}  "
              f"frontier log-MSE={r['frontier_mse']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
