#!/usr/bin/env python3
"""random selection: model-free vs. GP-infill recovery.

Both arms use ``--selection random`` (identical uniform picks per seed) and differ only in the
recovery route: GP grid-infill (``--model_type gp``) vs. model-free (``--model_type none``, fit
L(C) straight on the observed compute-optimal points). Isolates whether the surrogate infill helps.

    uv run python examples/random/compare_random.py --seeds 0-9
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.utils.plotting import run_comparison, METHOD_PALETTE  # noqa: E402

OUT = REPO / "results" / "baselines" / "random"
# label -> color + driver flags; both select randomly, differ only in the recovery model
EXPERIMENTS = {
    "random (GP infill)":  {"color": METHOD_PALETTE[0], "selection": "random", "model_type": "gp"},
    "random (model-free)": {"color": METHOD_PALETTE[1], "selection": "random", "model_type": "none"},
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
        out_path=OUT / "gp_vs_model_free.png",
        title=f"random: model-free vs. GP-infill recovery on {Path(args.input).stem}  "
              f"({len(seeds)} seeds, band = IQR; dashed = full-grid ground truth)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
