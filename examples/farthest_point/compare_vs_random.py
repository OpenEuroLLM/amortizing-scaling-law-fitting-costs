#!/usr/bin/env python3
"""farthest_point vs. random acquisition, through the canonical iterative_fits driver.

Runs ``iterative_fits_L0_form.py`` with ``--selection random`` and ``--selection farthest_point``
over several seeds, then reports how well each recovers the held-out scaling law as compute is
spent. farthest_point buys the pool point farthest (in standardized log-feature space) from the
observed set -- surrogate-free, space-filling; random draws uniformly. Both use the same driver,
the same GP surrogate for the *law fit* (only the acquisition differs), and the same shared init.

Figure panels: recovered law L(C)=L0+A*C^alpha vs. the true compute-optimal frontier (with the
held-out largest-N points marked) · held-out recovery MSE · the three recovered coefficients
(L0, A, alpha) vs. their full-grid ground truth.

    uv run python examples/farthest_point/compare_vs_random.py --seeds 0-4
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from iterative_fits_L0_form import data_preprocess, build_fit, functional_form_L0, INPUT_X  # noqa: E402
from scaling.utils import get_pareto_frontier  # noqa: E402

DRIVER = REPO / "iterative_fits_L0_form.py"
OUT = REPO / "results" / "baselines" / "farthest_point"
METHODS = {"random": ("#9ca3af", "random"), "farthest_point": ("#059669", "farthest_point")}


def clean_grid(src: Path) -> Path:
    """Driver-ready copy: rename bs->gbsz and cast N,D to float (avoids int64 C=6ND overflow)."""
    df = pd.read_csv(src).rename(columns={"bs": "gbsz"})
    df = df[["N", "D", "lr", "gbsz", "loss"]].dropna()
    df[["N", "D"]] = df[["N", "D"]].astype(float)
    dst = src.with_name(src.stem + "_clean.csv")
    df.to_csv(dst, index=False)
    return dst


def run_driver(grid: Path, selection: str, seed: int, niters: int, init_points: int):
    """Run one driver seed; return its history (or None if the driver's parametric fit failed on
    this seed -- a pre-existing driver fragility where build_fit returns None and it crashes)."""
    out = OUT / "runs" / selection
    r = subprocess.run([sys.executable, str(DRIVER), "--input", str(grid), "--output_dir", str(out),
                        "--seed", str(seed), "--niters", str(niters), "--init_points", str(init_points),
                        "--selection", selection, "--filename", "h"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    parquet = out / f"seed_{seed}" / "h.parquet"
    if r.returncode != 0 or not parquet.exists():
        print(f"  [skip] {selection} seed {seed}: driver failed (parametric fit returned None)")
        return None
    h = pd.read_parquet(parquet)
    P = np.array([list(p) for p in h["params"]])            # [l0, a, alpha] per iteration
    return h.assign(cL0=np.exp(P[:, 0]), cA=np.exp(P[:, 1]), cAlpha=P[:, 2])


def band(hists, key, absval=False):
    k = min(len(h) for h in hists)
    H = [h.iloc[:k] for h in hists]
    x = np.vstack([h["total_compute_spent"] for h in H]).mean(0)
    Y = np.vstack([h[key] for h in H])
    if absval:
        Y = np.abs(Y)
    return x, np.nanmedian(Y, 0), np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(REPO / "data" / "steplaw.csv"))
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--niters", type=int, default=24)
    p.add_argument("--init_points", type=int, default=10)
    args = p.parse_args()
    lo, hi = (args.seeds.split("-") + [None])[:2]
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(x) for x in args.seeds.split(",")]

    grid = clean_grid(Path(args.input))
    runs = {m: [h for s in seeds if (h := run_driver(grid, m, s, args.niters, args.init_points)) is not None]
            for m in METHODS}
    for m in METHODS:
        if not runs[m]:
            raise RuntimeError(f"all seeds failed for {m}")

    # ground truth + reference points (same construction the driver uses internally)
    df = pd.read_csv(grid)
    df["C"] = 6.0 * df["N"] * df["D"]                       # float N,D -> no int64 overflow
    train_df, test_known, _ = data_preprocess(df.copy(), held_out_type="N")
    gt = build_fit(pd.concat([train_df, test_known])[INPUT_X + ["C", "loss"]])[0]   # [l0, a, alpha]
    gtL0, gtA, gtAlpha = np.exp(gt[0]), np.exp(gt[1]), gt[2]
    front_obs = get_pareto_frontier(train_df, x_name="C", y_name="loss")            # true frontier (pool)
    front_held = get_pareto_frontier(test_known, x_name="C", y_name="loss")         # held-out largest N

    OUT.mkdir(parents=True, exist_ok=True)
    Cgrid = np.geomspace(df["C"].min(), df["C"].max(), 200)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"farthest_point vs. random on {Path(args.input).stem}  "
                 f"({len(seeds)} seeds, band = IQR; dashed = full-grid ground truth)", fontweight="bold")

    # --- panel: recovered law + true frontier + held-out points ---
    ax = axes[0, 0]
    ax.scatter(train_df["C"], train_df["loss"], s=6, color="#d1d5db", alpha=0.5, zorder=1,
               label="all configs (per N,D scale)")
    ax.scatter(test_known["C"], test_known["loss"], s=10, color="#fca5a5", alpha=0.5, zorder=2)
    ax.scatter(front_obs["C"], front_obs["loss"], s=16, color="#111827", zorder=3, label="compute-optimal (pool)")
    ax.scatter(front_held["C"], front_held["loss"], s=70, marker="X", color="#dc2626", zorder=5,
               edgecolor="white", linewidth=0.6, label="held-out frontier (largest N)")
    ax.plot(Cgrid, functional_form_L0(Cgrid, gt, return_log_loss=False), "--", color="#111827",
            lw=2.0, zorder=4, label="full-grid fit")
    for m, (col, lab) in METHODS.items():
        fp = np.median([h[["cL0", "cA", "cAlpha"]].iloc[-1].to_numpy() for h in runs[m]], axis=0)
        ax.plot(Cgrid, functional_form_L0(Cgrid, [np.log(fp[0]), np.log(fp[1]), fp[2]], return_log_loss=False),
                color=col, lw=1.8, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("compute C (FLOPs)"); ax.set_ylabel("loss L")
    ax.set_title("recovered law + held-out points"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    # --- trajectory panels ---
    def traj(ax, key, title, ylab, gt_line=None, logy=True, absval=False):
        for m, (col, lab) in METHODS.items():
            x, med, q1, q3 = band(runs[m], key, absval=absval)
            ax.fill_between(x, q1, q3, color=col, alpha=0.15, lw=0)
            ax.plot(x, med, "-o", color=col, ms=3, label=lab)
        if gt_line is not None:
            ax.axhline(gt_line, ls="--", color="#111827", lw=1.3, label=f"full-grid={gt_line:.3g}")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("compute spent (FLOPs)"); ax.set_ylabel(ylab)
        ax.set_title(title); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    traj(axes[0, 1], "known_test_mse", "held-out recovery", "known-test MSE")
    traj(axes[0, 2], "regret_alpha", "exponent error", "|alpha regret|", absval=True)
    traj(axes[1, 0], "cL0", "coefficient  L0", "L0", gt_line=gtL0)
    traj(axes[1, 1], "cA", "coefficient  A", "A", gt_line=gtA)
    traj(axes[1, 2], "cAlpha", "coefficient  alpha (exponent)", "alpha", gt_line=gtAlpha, logy=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = OUT / "farthest_point_vs_random.png"
    fig.savefig(figpath, dpi=200, bbox_inches="tight")

    print(f"\nfull-grid ground truth:  L0={gtL0:.4g}  A={gtA:.4g}  alpha={gtAlpha:.4g}")
    print(f"  {'method':16s} {'known_test_mse':>14s} {'|alpha regret|':>14s} {'L0':>8s} {'A':>10s} {'alpha':>9s}")
    for m, (_, lab) in METHODS.items():
        hs = runs[m]
        rec = np.nanmedian([h["known_test_mse"].iloc[-1] for h in hs])
        reg = np.nanmedian([abs(h["regret_alpha"].iloc[-1]) for h in hs])
        fL0 = np.nanmedian([h["cL0"].iloc[-1] for h in hs]); fA = np.nanmedian([h["cA"].iloc[-1] for h in hs])
        fAl = np.nanmedian([h["cAlpha"].iloc[-1] for h in hs])
        print(f"  {lab:16s} {rec:14.4e} {reg:14.4e} {fL0:8.3f} {fA:10.3g} {fAl:9.4f}")
    print("saved", figpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
