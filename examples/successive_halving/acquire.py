#!/usr/bin/env python3
"""SH-as-GP-acquisition vs. random, on the real steplaw grid.

Both strategies acquire real ``(N, D, lr, gbsz)`` grid rows one at a time; after
each pick we fit ``L(C)=E+A*C^b`` on the observed loss-compute frontier and score
it on the held-out largest-N models. ``sh_gp`` (``SuccessiveHalvingGP``) uses the
loop's GP to advance the most promising size's curve and pick its config; the GP
also chooses the hyperparameters, so HP cost is paid like the random baseline.

    uv run python examples/successive_halving/acquire.py --seeds 0-9

Caveat: steplaw has only 5 model sizes (4 after holding out the largest) and 2-4
D per size, so SH is shallow here -- this is an honest illustration on real data,
not the paper's dense regime. See README.
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))                              # for `selection`, `models`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling modules
from core.selection import FEATURES, SuccessiveHalvingGP

OUT = REPO / "results" / "baselines" / "successive_halving"


class GP:
    """Robust GP over (log-)features -> loss (input Normalize + outcome Standardize).

    Handles the few-observation regime that a bare SingleTaskGP fails on; exposes a
    ``predict`` so it drops into ``SuccessiveHalvingGP.next_point`` as the surrogate.
    """

    def __init__(self, X, y, seed=None):
        import botorch
        import gpytorch
        import torch
        from botorch.models.transforms import Normalize, Standardize

        self._torch = torch
        if seed is not None:
            torch.manual_seed(seed)          # make botorch's random restarts reproducible
        Xt, yt = torch.tensor(X), torch.tensor(y)
        self._gp = botorch.models.SingleTaskGP(
            Xt, yt, input_transform=Normalize(X.shape[1]), outcome_transform=Standardize(1))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._gp.likelihood, self._gp)
        botorch.fit_gpytorch_mll(mll)
        self._gp.eval()

    def predict(self, X):
        with self._torch.no_grad():
            return self._gp.posterior(self._torch.tensor(X)).mean.numpy().ravel()


def load_grid(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"bs": "gbsz"})
    df[["N", "D"]] = df[["N", "D"]].astype(float)
    df["C"] = 6.0 * df["N"] * df["D"]
    return df[FEATURES + ["C", "loss"]].reset_index(drop=True)


def cell_opt(df):
    """Compute-optimal point per (N, D) cell: the min-loss config there."""
    return df.loc[df.groupby(["N", "D"])["loss"].idxmin()]


def frontier_rows(df):
    """Loss-compute frontier rows: at each compute the min loss, non-increasing envelope."""
    d = df.sort_values(["C", "loss"])
    keep = d["loss"].to_numpy() <= np.minimum.accumulate(d["loss"].to_numpy()) + 1e-12
    return d[keep]


# --- law forms: fit(df)->coeffs dict, predict(coeffs, df)->loss. Fitted in log-loss space,
#     multi-start over exponent seeds, so the ill-conditioned few-point fits stay sane. ------
def _multistart(model, X, ylog, p0s, bounds):
    best, best_sse = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p0 in p0s:
            try:
                popt, _ = curve_fit(model, X, ylog, p0=p0, bounds=bounds, maxfev=40000)
            except Exception:
                continue
            sse = float(np.sum((model(X, *popt) - ylog) ** 2))
            if sse < best_sse:
                best, best_sse = popt, sse
    if best is None:
        raise RuntimeError("fit failed")
    return best


def fit_lc_l0(df):
    C, y, Emax = df["C"].to_numpy(), np.log(df["loss"].to_numpy()), df["loss"].min()
    m = lambda C, E, lA, b: np.log(E + np.exp(lA) * C ** b)
    p0s = [[e, np.log(1e3), b] for e in (0.5, 1.0, 1.5) for b in (-0.05, -0.1, -0.2)]
    p = _multistart(m, C, y, p0s, ([0, -20, -1], [Emax, 40, 0]))
    return {"E": p[0], "A": float(np.exp(p[1])), "b": p[2]}


def fit_lc_power(df):
    C, y = np.log(df["C"].to_numpy()), np.log(df["loss"].to_numpy())
    m = lambda lC, lA, b: lA + b * lC
    p = _multistart(m, C, y, [[np.log(1e3), -0.1]], ([-20, -1], [40, 0]))
    return {"A": float(np.exp(p[0])), "b": p[1]}


def fit_nd(df):
    N, D = df["N"].to_numpy(), df["D"].to_numpy()
    y, Emax = np.log(df["loss"].to_numpy()), df["loss"].min()
    X = np.vstack([N, D])
    m = lambda X, E, lA, al, lB, be: np.log(E + np.exp(lA) * X[0] ** -al + np.exp(lB) * X[1] ** -be)
    p0s = [[Emax * 0.7, np.log(400), al, np.log(400), be]
           for al in (0.1, 0.3, 0.5) for be in (0.1, 0.3, 0.5)]
    p = _multistart(m, X, y, p0s, ([0, -5, 0.01, -5, 0.01], [Emax, 15, 1, 15, 1]))
    return {"E": p[0], "A": float(np.exp(p[1])), "alpha": p[2], "B": float(np.exp(p[3])), "beta": p[4]}


def pred_lc(c, df):
    return c.get("E", 0.0) + c["A"] * df["C"].to_numpy() ** c["b"]


def pred_nd(c, df):
    return c["E"] + c["A"] * df["N"].to_numpy() ** -c["alpha"] + c["B"] * df["D"].to_numpy() ** -c["beta"]


# ``points`` = which observed rows a form is fit on: L(C) uses the loss-compute frontier
# (envelope); L(N,D) uses every compute-optimal (N,D) cell (each is a distinct data point).
LAWS = {
    "lc_l0":    dict(names=["E", "A", "b"],                  fit=fit_lc_l0,    pred=pred_lc, logy={"A"},      points=frontier_rows),
    "lc_power": dict(names=["A", "b"],                       fit=fit_lc_power, pred=pred_lc, logy={"A"},      points=frontier_rows),
    "nd":       dict(names=["E", "A", "alpha", "B", "beta"], fit=fit_nd,       pred=pred_nd, logy={"A", "B"}, points=cell_opt),
}


def law_curve(name, c, Cgrid, N_lo, N_hi):
    """Compute-optimal loss vs compute implied by fitted coeffs (log-log a near-line).

    L(C) forms are already a function of C; for L(N,D) we take the compute-optimal
    envelope min_N L(N, C/6N) over a dense grid of sizes.
    """
    if name in ("lc_l0", "lc_power"):
        return c.get("E", 0.0) + c["A"] * Cgrid ** c["b"]
    Ng = np.geomspace(N_lo, N_hi, 300)[None, :]
    D = Cgrid[:, None] / (6.0 * Ng)
    return (c["E"] + c["A"] * Ng ** -c["alpha"] + c["B"] * D ** -c["beta"]).min(axis=1)


def init_points(train, seed):
    """Shared start: one cheapest-D row per model size (a fair, method-agnostic seed)."""
    rng = np.random.RandomState(seed)
    idx = []
    for N, sub in train.groupby("N"):
        cell = sub[sub.D == sub.D.min()]
        idx.append(int(rng.choice(cell.index)))
    return pd.Index(idx)


def run(method, train, ref, law, seed, niters):
    """Acquire rows; each step fit the chosen law form on the observed compute-optimal
    points and score how well it recovers the full-grid law over the explored compute."""
    spec = LAWS[law]
    acquired = init_points(train, seed)
    sh = SuccessiveHalvingGP(seed=seed, explore=0.5 if method == "sh_gp_explore" else 0.0)
    rng = np.random.RandomState(1000 + seed)
    hist = []
    for it in range(niters + 1):
        obs = train.loc[acquired]
        gp = None
        if method in ("sh_gp", "sh_gp_explore"):
            try:
                gp = GP(np.log(obs[FEATURES].to_numpy(float)), obs["loss"].to_numpy()[:, None],
                        seed=seed * 10_000 + it)   # reproducible per (run seed, iteration)
            except Exception:
                gp = None                        # fall back to random until the GP fits
        try:
            coeffs = spec["fit"](spec["points"](obs))
            lo, hi = obs["C"].min(), obs["C"].max()
            refm = ref[(ref["C"] >= lo) & (ref["C"] <= hi)]      # recover over what we explored
            rec = (float(np.mean((np.log(spec["pred"](coeffs, refm)) - np.log(refm["loss"])) ** 2))
                   if len(refm) >= 2 else np.nan)
        except Exception:
            coeffs, rec = {n: np.nan for n in spec["names"]}, np.nan
        hist.append(dict(n=len(acquired), compute=float(obs["C"].sum()), recovery=rec,
                         min_loss=float(obs["loss"].min()), **coeffs))
        if it == niters:
            break
        cand = train.index.difference(acquired)
        if method == "random":
            new = int(rng.choice(cand))
        else:
            new = sh.next_point(train, cand, acquired_idx=acquired, surrogate=gp)
        acquired = acquired.append(pd.Index([new]))
    return pd.DataFrame(hist)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", default=str(REPO / "data" / "steplaw.csv"),
                   help="Path to a grid CSV with N, D, lr, gbsz (or bs), loss.")
    p.add_argument("--law", default="nd", choices=list(LAWS),
                   help="Law form to fit/report: nd = E+A*N^-a+B*D^-b, lc_l0 = E+A*C^b, lc_power = A*C^b.")
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--niters", type=int, default=24)
    args = p.parse_args()
    lo, hi = (args.seeds.split("-") + [None])[:2]
    seeds = range(int(lo), int(hi) + 1) if hi else [int(x) for x in args.seeds.split(",")]
    spec = LAWS[args.law]

    df = load_grid(Path(args.grid))
    tag = Path(args.grid).stem
    train = df[df.N < df.N.max()].copy()
    ref = spec["points"](df)                         # full-grid points for this form
    gt = spec["fit"](ref)                            # ground truth = full-grid fit

    runs = {m: [run(m, train, ref, args.law, s, args.niters) for s in seeds]
            for m in ("sh_gp", "sh_gp_explore", "random")}
    # each method's final recovered law (median coeffs over seeds), for the law-vs-compute panel
    fin = {m: {k: float(np.nanmedian([h[k].iloc[-1] for h in hs])) for k in spec["names"]}
           for m, hs in runs.items()}
    best_loss = float(train["loss"].min())           # lowest loss any model in the pool reaches

    out = OUT / f"acquire_{tag}"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    col = {"sh_gp": "#2563eb", "sh_gp_explore": "#ea580c", "random": "#9ca3af"}
    lab = {"sh_gp": "SH-GP", "sh_gp_explore": "SH-GP + explore", "random": "Random"}
    Cgrid = np.geomspace(ref["C"].min(), ref["C"].max(), 200)
    N_lo, N_hi = float(df["N"].min()), float(df["N"].max())

    panels = ["law", "recovery", "min_loss"] + spec["names"]
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    axes = axes.ravel()
    fig.suptitle(f"SH-as-GP-acquisition vs. random on {tag}  —  law: {args.law}  "
                 f"(dashed = full-grid ground truth)", fontweight="bold")

    def med(hs, key):
        k = min(len(h) for h in hs)
        H = [h.iloc[:k] for h in hs]
        x = np.vstack([h["compute"] for h in H]).mean(0)
        return x, np.nanmedian(np.vstack([h[key] for h in H]), 0)

    for ax, key in zip(axes, panels):
        if key == "law":
            # the actual recovered scaling law vs compute, over the compute-optimal points
            ax.scatter(ref["C"], ref["loss"], s=12, color="#111827", zorder=2,
                       label="compute-optimal points (truth)")
            ax.plot(Cgrid, law_curve(args.law, gt, Cgrid, N_lo, N_hi), "--", color="#059669",
                    lw=2.2, zorder=4, label="ground-truth law")
            for m in runs:
                ax.plot(Cgrid, law_curve(args.law, fin[m], Cgrid, N_lo, N_hi), color=col[m],
                        lw=1.8, zorder=3, label=lab[m])
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("compute C (FLOPs)"); ax.set_ylabel("loss L")
            ax.set_title("recovered law vs compute-optimal frontier")
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
            continue
        for m, hs in runs.items():
            x, y = med(hs, key)
            ax.plot(x, y, "-o", color=col[m], ms=3, label=lab[m])
        ax.set_xscale("log"); ax.set_xlabel("compute spent (FLOPs)"); ax.grid(alpha=0.3, which="both")
        if key == "recovery":
            ax.set_yscale("log"); ax.set_ylabel("frontier recovery (log-MSE)"); ax.set_title("recovery")
        elif key == "min_loss":
            ax.axhline(best_loss, ls="--", color="#111827", lw=1.3, label=f"best model={best_loss:.3f}")
            ax.set_ylabel("lowest loss reached  (l*)")
            ax.set_title("l*  (SH's actual objective)")
        else:
            ax.axhline(gt[key], ls="--", color="#111827", lw=1.3, label=f"ground truth={gt[key]:.3g}")
            if key in spec["logy"]:
                ax.set_yscale("log")
            ax.set_title(f"coefficient  {key}")
        ax.legend(fontsize=8)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = out / "figures" / f"sh_gp_vs_random_{args.law}.png"  # under acquire_<grid>/
    fig.savefig(figpath, dpi=200, bbox_inches="tight")

    print(f"law form: {args.law}   ground-truth (full-grid) coeffs:")
    print("  " + "  ".join(f"{k}={v:.4g}" for k, v in gt.items()))
    for m, hs in runs.items():
        fin = {k: np.nanmedian([h[k].iloc[-1] for h in hs]) for k in spec["names"]}
        rec = np.nanmedian([h["recovery"].iloc[-1] for h in hs])
        print(f"  {lab[m]:20s} final median: recovery={rec:.3e}  "
              + "  ".join(f"{k}={v:.4g}" for k, v in fin.items()))
    print("saved", figpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
