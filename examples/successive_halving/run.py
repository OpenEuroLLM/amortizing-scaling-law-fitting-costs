#!/usr/bin/env python3
"""Run uniform / SH / SH-surrogate over many seeds and report the comparison.

Reproduces the paper's headline: strategic allocation (SH) beats uniform, and
SH with a surrogate beats plain SH -- both on the loss it reaches and on the
scaling law it recovers.

    uv run python examples/successive_halving/run.py
    uv run python examples/successive_halving/run.py --n-models 20 --budget 1e21 --seeds 0-49
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))                      # for `selection`, `models`
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for sibling modules
from selection import SuccessiveHalving
from fit_eval import evaluate
from surrogate import gp_surrogate
from synth_lc import loss_at_compute, sample_models

OUT = REPO / "results" / "baselines" / "successive_halving"


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def one_seed(seed, n_models, budget, eta, noise):
    models = sample_models(n_models, seed)
    sh = SuccessiveHalving(seed=seed, eta=eta)
    runs = {
        "uniform": sh.uniform(models, budget, loss_at_compute),
        "SH": sh.run(models, budget, loss_at_compute, noise=noise),
        "SH-surrogate": sh.run(models, budget, loss_at_compute, surrogate=gp_surrogate, noise=noise),
    }
    return {m: evaluate(r) for m, r in runs.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-models", type=int, default=16)
    p.add_argument("--budget", type=float, default=1e20, help="Total compute B in FLOPs.")
    p.add_argument("--eta", type=float, default=2.0, help="SH pruning factor (keep top 1/eta).")
    p.add_argument("--noise", type=float, default=0.3,
                   help="Std of Gaussian noise on the observed loss (the surrogate's edge shows here).")
    p.add_argument("--seeds", default="0-19", help="e.g. '0-49' or '0,1,2'.")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    rows = []
    for seed in seeds:
        ev = one_seed(seed, args.n_models, args.budget, args.eta, args.noise)
        for m, r in ev.items():
            E, A, b = r["law"]
            rows.append(dict(seed=seed, method=m, l_star=r["l_star"], spent=r["spent"],
                             frontier_mse=r["frontier_mse"], law_E=E, law_A=A, law_b=b))
        print(f"seed {seed:3d}: "
              + "  ".join(f"{m} l*={ev[m]['l_star']:.4f}" for m in ev))
    df = pd.DataFrame(rows)

    tag = f"M{args.n_models}_B{args.budget:.0e}_eta{args.eta:g}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    # paper-style aggregate: relative improvement over plain SH (per seed, then mean/max).
    piv = df.pivot(index="seed", columns="method", values="l_star")
    rel_surr = (piv["SH"] - piv["SH-surrogate"]) / piv["SH"]        # >0 : surrogate better
    rel_unif = (piv["uniform"] - piv["SH"]) / piv["SH"]             # >0 : uniform worse (SH better)
    print(f"\n=== {tag}  ({len(seeds)} seeds) ===")
    print(f"mean l*        uniform={piv['uniform'].mean():.4f}  "
          f"SH={piv['SH'].mean():.4f}  SH-surrogate={piv['SH-surrogate'].mean():.4f}")
    print(f"SH-surrogate vs SH : mean rel. improvement {rel_surr.mean()*100:+.2f}%  "
          f"(max {rel_surr.max()*100:+.2f}%)")
    print(f"uniform      vs SH : mean rel. degradation {rel_unif.mean()*100:+.2f}%  "
          f"(max {rel_unif.max()*100:+.2f}%)")
    print(f"mean frontier log-MSE  "
          + "  ".join(f"{m}={df[df.method==m].frontier_mse.mean():.3e}" for m in piv.columns))
    print(f"\nsaved {out/'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
