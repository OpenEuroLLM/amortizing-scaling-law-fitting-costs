#!/usr/bin/env python3
"""Overlay N experiments on one comparison figure.

A general template for comparing 2-7 configurations at once. Edit ``EXPERIMENTS`` below: each row is
``label -> {"color": ..., <driver_flag>: <value>, ...}`` where the flags are any ``main.py loss_law``
option (``selection``, ``model_type``, ``feature_set``, ``form``, ...). Put the baseline in
METHOD_PALETTE[0] (a recessive gray) and each treatment in [1], [2], ... . Keep ONE axis of variation
per figure so a difference is attributable (vary selection OR feature_set OR model_type, not several).

    uv run python examples/plot_compare.py --seeds 0-9                 # loss-law (L0) comparison
    uv run python examples/plot_compare.py --task hp_law --seeds 0-9   # optimal-HP (step-law) comparison

The default below compares four selection/recovery configurations. ``--task hp_law`` reuses the same
rows against the optimal-HP pipeline (panel 0 becomes optimal-HP-vs-scale). Two more ready-to-use
sets are shown commented out.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core.utils.plotting import run_comparison, METHOD_PALETTE  # noqa: E402

OUT = REPO / "results" / "baselines" / "compare"

# --- edit these rows to compare anything ---
EXPERIMENTS = {
    "random (model-free)": {"color": METHOD_PALETTE[0], "selection": "random", "model_type": "none"},
    "random (GP)":         {"color": METHOD_PALETTE[1], "selection": "random", "model_type": "gp"},
    "max_variance (GP)":   {"color": METHOD_PALETTE[2], "selection": "max_variance", "model_type": "gp"},
    "farthest_point (GP)": {"color": METHOD_PALETTE[3], "selection": "farthest_point", "model_type": "gp"},
}

# Feature-set sweep (needs an --extended grid carrying width/depth, e.g. a steplaw CSV with them):
# EXPERIMENTS = {
#     "default": {"color": METHOD_PALETTE[0], "feature_set": "default"},
#     "arch":    {"color": METHOD_PALETTE[1], "feature_set": "arch"},
#     "data":    {"color": METHOD_PALETTE[2], "feature_set": "data"},
#     "full":    {"color": METHOD_PALETTE[3], "feature_set": "full"},
# }


def parse_seeds(s):
    lo, hi = (s.split("-") + [None])[:2]
    return list(range(int(lo), int(hi) + 1)) if hi else [int(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(REPO / "data" / "steplaw.csv"))
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--task", default="loss_law", choices=["loss_law", "hp_law"])
    p.add_argument("--niters", type=int, default=24)
    p.add_argument("--init_points", type=int, default=10)
    p.add_argument("--form", default=None, help="fit form (default: L0 for loss_law, steplaw_lr for hp_law)")
    p.add_argument("--out", default=str(OUT / "compare.png"), help="output PNG path")
    p.add_argument("--error_param", default="alpha",
                   help="param whose |regret| gets the headline panel (skipped if the form lacks it)")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    run_comparison(
        EXPERIMENTS, args.input, seeds, task=args.task, form_name=args.form,
        niters=args.niters, init_points=args.init_points,
        out_path=Path(args.out), error_param=args.error_param,
        title=f"{' vs. '.join(EXPERIMENTS)} on {Path(args.input).stem}  "
              f"({len(seeds)} seeds, band = IQR; dashed = full-grid ground truth)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
