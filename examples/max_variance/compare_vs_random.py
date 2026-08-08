#!/usr/bin/env python3
"""max_variance vs. random acquisition, through the loss_law pipeline.

Both arms use the same GP surrogate for the law fit and the same shared init; they differ only in
*which point they buy* -- max_variance takes the pool point of highest GP posterior variance
(uncertainty sampling), random draws uniformly.

    uv run python examples/max_variance/compare_vs_random.py --seeds 0-4
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.utils.plotting import run_comparison, METHOD_PALETTE  # noqa: E402

OUT = REPO / "results" / "baselines" / "max_variance"
EXPERIMENTS = {
    "random":            {"color": METHOD_PALETTE[0], "selection": "random"},
    "max_variance (GP)": {"color": METHOD_PALETTE[1], "selection": "max_variance"},
}


def parse_seeds(s):
    lo, hi = (s.split("-") + [None])[:2]
    return list(range(int(lo), int(hi) + 1)) if hi else [int(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(REPO / "data" / "steplaw.csv"))
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--niters", type=int, default=24)
    p.add_argument("--init_points", type=int, default=10)
    p.add_argument("--form", default="L0")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    run_comparison(
        EXPERIMENTS, args.input, seeds, form_name=args.form,
        niters=args.niters, init_points=args.init_points,
        out_path=OUT / "max_variance_vs_random.png",
        title=f"max_variance vs. random on {Path(args.input).stem}  "
              f"({len(seeds)} seeds, band = IQR; dashed = full-grid ground truth)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
