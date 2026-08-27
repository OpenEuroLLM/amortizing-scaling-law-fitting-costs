#!/usr/bin/env python3
r"""L(C) fit-across-compute figure for the L0 form, rendered from a finished loss_law run's sidecars
(no re-run needed). Reproduces the pipeline's L-vs-C "real vs fantasized fit" panel
(core/pipelines/loss_law.py:_plot): loss vs FLOPs, all observed points (small grey), the observed
compute-optimal frontier (black), held-out eval points (blue X), the held-out compute region shaded,
and three fitted L(C) = L0 + A*C^alpha curves -- subsampled grid (obs-only, green), fantasized dense
grid (obs+fant, red), and full pool fit (all acquirable-pool points, purple dash-dot) -- each solid
within the seen C range and dashed beyond it. No curve sees held-out data: the held-out region is a
true extrapolation target for all three fits.

Reads from a run directory: history.parquet (per-iter `params` = obs+fant, `obs_only_params`,
`acquired_idx`) + ref_fits.json (`validation` = reference fit on the acquirable pool only). The
pool/held split is recomputed the same way the pipeline does it.

Usage (from repo root):
    python experiments/workshop_result/analysis/plot_fits_across_C.py \
        --run "experiments/workshop_result/L(C)/runs/bo_ei/oellm_english/L0/seed_0" \
        --iter 60 --out experiments/workshop_result/analysis/out/fig_L0_fits_across_C.pdf
If --run is a parent dir, the latest timestamped run under it is used. The L(C) runs come from
`../L(C)/run.sh` (same L0/EI recipe the bo_ei_recovery study used).
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

import sys
REPO = Path(__file__).resolve()
while not (REPO / "main.py").exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
from core.forms import PARAMETRIC_FORMS
from core.utils.data import split_search_space
from scaling.utils import get_pareto_frontier
from core.utils.plotting import set_pub_style

TEAL, BLUE_X = "#008969", "#265ddc"


def resolve_run(p: Path) -> Path:
    if (p / "history.parquet").exists():
        return p
    hits = sorted(glob.glob(str(p / "**" / "history.parquet"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"no history.parquet under {p}")
    return Path(hits[-1]).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir (or a parent; latest run is used)")
    ap.add_argument("--input", default="data/oellm_english/preprocessed/oellm_english.csv")
    ap.add_argument("--form", default="L0")
    ap.add_argument("--split_axis", default="C")
    ap.add_argument("--held_out_frac", type=float, default=0.5)
    ap.add_argument("--iter", type=int, default=None,
                    help="history iteration to snapshot (default: ~1/3 of the way through)")
    ap.add_argument("--ext_decades", type=float, default=1.5, help="decades to extrapolate past the grid")
    ap.add_argument("--out", default="experiments/workshop_result/analysis/out/fig_L0_fits_across_C.pdf")
    args = ap.parse_args()

    run = resolve_run(Path(args.run))
    form = PARAMETRIC_FORMS[args.form]
    hist = pd.read_parquet(run / "history.parquet")
    ref = json.loads((run / "ref_fits.json").read_text())
    pool_params = np.asarray(ref["validation"]["params"], float)    # reference fit on the ACQUIRABLE POOL
                                                                    # (no fit here ever sees held-out data)

    df = pd.read_csv(REPO / args.input); df["C"] = 6.0 * df["N"] * df["D"]
    pool_df, held_df = split_search_space(df.copy(), held_out_frac=args.held_out_frac, axis=args.split_axis)
    combined = pd.concat([pool_df, held_df])
    glob_par = get_pareto_frontier(combined, x_name="C", y_name="loss")
    held_front = glob_par.loc[glob_par.index.isin(held_df.index)].sort_values("C")   # held-out eval pts

    it = args.iter if args.iter is not None else int(hist["iter"].iloc[len(hist) // 3])
    row = hist.iloc[(hist["iter"] - it).abs().idxmin()]
    obs_idx = [int(i) for i in row["acquired_idx"]]
    obs_df = pool_df.loc[pool_df.index.intersection(obs_idx)].copy()
    obs_opt = get_pareto_frontier(obs_df, x_name="C", y_name="loss")         # observed compute-optimal
    fant_params = np.asarray(row["params"], float)
    obs_params = np.asarray(row["obs_only_params"], float)
    print(f"run={run.name} | iter={int(row['iter'])} | observed={len(obs_df)} pts "
          f"(C up to {obs_df['C'].max():.2e}) | held-out eval={len(held_front)} pts")

    # --- figure (matches the pipeline's L-vs-C panel: loss_law.py:_plot) ---
    set_pub_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    x_seen_max = float(obs_df["C"].max())
    x_split = float(held_df["C"].min())                          # pool | held-out boundary
    x_ext_max = float(held_front["C"].max()) * 10.0 ** args.ext_decades   # C_unknown extrapolation depth

    # held-out region: the narrow band spanning the held-out compute range
    ax.axvspan(x_split, float(held_df["C"].max()), color=TEAL, alpha=0.25, lw=0, label="held-out region")

    # observed points (small grey), observed compute-optimal frontier (black), held-out eval (blue X)
    ax.scatter(obs_df["C"], obs_df["loss"], marker="o", color="black", s=5, alpha=0.30, zorder=2,
               label="observed all points")
    ax.scatter(obs_opt["C"], obs_opt["loss"], marker="o", color="black", s=25, zorder=3,
               label="observed pareto")
    ax.scatter(held_front["C"], held_front["loss"], marker="x", color=BLUE_X, s=30, zorder=5,
               label="held-out eval")

    # three L(C) fits, solid within the seen C range and dashed beyond:
    #   subsampled grid   = obs-only fit (green)   | fantasized dense grid = obs+fant fit (red)
    #   full pool fit     = fit on the whole acquirable pool (purple dash-dot)
    C_grid = np.geomspace(float(obs_df["C"].min()), x_ext_max, 300)
    seen = C_grid <= x_seen_max
    for params, color, label in [(obs_params, "tab:green", "subsampled grid"),
                                 (fant_params, "red", "fantasized dense grid")]:
        y = form.to_L_vs_C(C_grid, params)
        ax.plot(C_grid[seen], y[seen], color=color, ls="-", lw=1.6, zorder=4, label=label)
        ax.plot(C_grid[~seen], y[~seen], color=color, ls="--", lw=1.6, zorder=4)
    yg = form.to_L_vs_C(C_grid, pool_params)
    ax.plot(C_grid[seen], yg[seen], color="tab:purple", ls="-.", lw=1.6, zorder=4, label="full pool fit")
    ax.plot(C_grid[~seen], yg[~seen], color="tab:purple", ls="--", lw=1.6, zorder=4)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FLOPs"); ax.set_ylabel("loss")
    ax.set_title("real vs fantasized fit (dashed = beyond seen C)")
    ax.set_xlim(float(obs_df["C"].min()) * 0.8, x_ext_max)
    ax.grid(which="both", ls="--", lw=1, alpha=0.5)
    ax.legend(fontsize=10, ncol=1, loc="upper right", framealpha=0.9)

    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main()
