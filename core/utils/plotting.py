"""Plotting styles, palettes, and the shared experiment-comparison figure.

Two layers:

- **Styling** -- ``set_pub_style`` (a publication rcParams preset) + palettes. Every figure calls
  ``set_pub_style`` so comparison figures and per-run diagnostics share one look.
- **Comparison figure** -- ``run_experiments`` drives the pipeline over seeds per config,
  ``ground_truth`` builds the full-grid references, and ``comparison_figure`` renders the canonical,
  form-generic 2xN figure (recovered law + held-out points, held-out MSE, optional headline error,
  and one coefficient-trajectory panel per interpretable parameter, median + IQR band per experiment).

The categorical method palette is the Okabe-Ito colorblind-safe set led by a neutral gray baseline
(validated with the dataviz skill's checker). The gray is deliberately achromatic -- it is the
recessive *reference* series, always carrying a legend label + line markers as secondary encoding, so
the treatment hue pops; that is why it sits below the palette's chroma floor by design.
"""
import subprocess
import sys
from math import ceil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

REPO = Path(__file__).resolve().parents[2]

# --- palettes ---

# Model-size (N) and token-budget (D) sequential palettes (per-scale coloring of raw points).
N_palette = ["#ff62e0", "#ab68d8", "#5d63bb", "#19548e", "#003f5c", "#162C36"]
D_palette = ["#10ffbf", "#23e4ac", "#2bca99", "#2fb186", "#309874", "#308063",
             "#2d6852", "#295242", "#243c32"]

# Categorical palette for overlaying experiments: gray baseline (recessive) + Okabe-Ito hues,
# ordered for maximum separation from the gray and from each other (dataviz-validated).
METHOD_PALETTE = ["#6b7280", "#D55E00", "#009E73", "#CC79A7", "#0072B2", "#E69F00", "#56B4E9"]

FORMAT_BILLIONS = lambda x: f"{float(x)/1e9:g}B"
FORMAT_N = lambda x: f"{float(x)/1e9:.2g}B"


def build_N_color_map(Ns):
    """Map each sorted, unique value of ``Ns`` to a color from ``N_palette`` (cycled past its length)."""
    Ns_sorted = sorted(set(float(n) for n in Ns))
    return {n: N_palette[i % len(N_palette)] for i, n in enumerate(Ns_sorted)}


def _sci_label(v, _pos=None):
    """m x 10^n with 2 sig figs; bare 10^n when m == 1 (clean log-tick labels)."""
    if v <= 0:
        return ""
    e = int(np.floor(np.log10(v)))
    m = v / 10.0 ** e
    return rf"$10^{{{e}}}$" if abs(m - 1) < 1e-6 else rf"${m:.2g}\times10^{{{e}}}$"


# --- publication style ---

_PUB_RCPARAMS = {
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.titlesize": 14, "figure.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.grid.which": "both", "axes.axisbelow": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.6,
    "lines.linewidth": 1.8, "lines.markersize": 4,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#d1d5db",
    "legend.fancybox": False,
}


def set_pub_style():
    """Apply the publication rcParams preset (idempotent). Call before building any figure."""
    mpl.rcParams.update(_PUB_RCPARAMS)


def tidy_log_axis(ax, which="x", label_fn=None):
    """Label a log axis at a few clean ticks (1/2/5 x 10^n in range; else ~4 log-spaced), no crowding.

    Call after the data is plotted (limits set). Robust to sub-decade ranges, where the usual
    1/2/5 marks can fall entirely outside the view and matplotlib would otherwise crowd the minors.
    ``label_fn`` formats a tick value to a string; defaults to scientific ``m x 10^n`` notation.
    """
    label_fn = label_fn or _sci_label
    a = ax.xaxis if which == "x" else ax.yaxis
    lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
    lo, hi = max(lo, hi * 1e-300), hi
    decades = list(range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1))
    fine = [m * 10.0 ** d for d in decades for m in (1, 2, 5) if lo <= m * 10.0 ** d <= hi]
    if len(fine) < 2:                       # narrow sub-decade range: a few log-spaced ticks
        ticks = list(np.geomspace(lo, hi, 4))
    elif len(fine) <= 8:                    # moderate range: 1/2/5 x 10^n reads cleanly
        ticks = fine
    else:                                   # very wide range: decade majors, thinned to <= ~9
        ticks = [10.0 ** d for d in decades if lo <= 10.0 ** d <= hi]
        if len(ticks) > 9:
            ticks = ticks[:: int(np.ceil(len(ticks) / 9))]
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _pos=None: label_fn(v)))
    a.set_minor_locator(NullLocator())


# --- median + IQR bands over seeds ---

def _band_series(xs, ys, absval=False):
    """(x, median, q25, q75) over seed-aligned series, truncated to the shortest; x = mean of xs."""
    k = min(len(y) for y in ys)
    X = np.vstack([np.asarray(x)[:k] for x in xs]).mean(0)
    Y = np.vstack([np.asarray(y)[:k] for y in ys])
    if absval:
        Y = np.abs(Y)
    return X, np.nanmedian(Y, 0), np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0)


def band(hists, key, absval=False):
    """Median + IQR of ``key`` vs ``total_compute_spent`` across a config's seed histories."""
    return _band_series([h["total_compute_spent"] for h in hists],
                        [h[key].to_numpy() for h in hists], absval)


# --- run experiments + build ground-truth references ---

def run_experiments(experiments, grid, seeds, *, out_dir, task="loss_law",
                    niters=24, init_points=10):
    """Drive ``main.py <task>`` over ``seeds`` for each experiment; return ``{label: [hist, ...]}``.

    ``experiments`` is an ordered dict ``label -> {"color": ..., <cli_flag>: <value>, ...}`` where the
    flags (everything but ``color``) are passed to the driver (e.g. ``selection``, ``model_type``,
    ``feature_set``, ``form``). Seeds whose driver run fails (a pre-existing fit fragility) are skipped
    with a warning; a config with no surviving seed raises.
    """
    from argparse import Namespace

    from core.pipelines._common import find_latest_run_dir

    driver = REPO / "main.py"
    out_dir = Path(out_dir)
    results = {}
    for i, (label, spec) in enumerate(experiments.items()):
        flags = {k: v for k, v in spec.items() if k != "color"}
        key = "_".join([task] + [f"{k}-{v}" for k, v in flags.items()]) or f"exp{i}"
        hists = []
        for s in seeds:
            out = out_dir / "runs" / key
            cmd = [sys.executable, str(driver), task, "--input", str(grid), "--output_dir", str(out),
                   "--seed", str(s), "--niters", str(niters), "--init_points", str(init_points),
                   "--filename", "h"]
            for k, v in flags.items():
                cmd += [f"--{k}", str(v)]
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            run_args = Namespace(
                output_dir=str(out),
                input=str(grid),
                form=flags["form"],
                seed=s,
                filename="h",
            )
            run_dir = find_latest_run_dir(run_args)
            parquet = run_dir / "history.parquet" if run_dir else None
            if r.returncode == 0 and parquet is not None and parquet.exists():
                hists.append(pd.read_parquet(parquet))
            else:
                print(f"  [skip] {label} seed {s}: driver failed (parametric fit returned None)")
        if not hists:
            raise RuntimeError(f"all seeds failed for {label}")
        results[label] = hists
    return results


def ground_truth(grid_df, form, held_out_frac=0.2, axis="C"):
    """Full-grid references for the recovered-law panel: fit + the two regions' frontiers + a grid.

    Uses the same ``split_search_space`` split the pipeline runs, so the panel's pool / held-out
    markers match what the runs scored (defaults mirror the pipeline defaults).
    """
    from scaling.utils import get_pareto_frontier
    from core.utils.data import split_search_space
    from core.fitting import build_fit

    df = grid_df.copy()
    df["C"] = 6.0 * df["N"] * df["D"]                       # float N,D -> no int64 overflow
    pool_df, held_out_df = split_search_space(df, held_out_frac, axis)
    gt_params = build_fit(df, form_name=form.name, y_name="loss")[0]
    return {
        "params": gt_params,
        "train_df": pool_df,
        "test_known": held_out_df,
        "front_obs": get_pareto_frontier(pool_df, x_name="C", y_name="loss"),
        "front_held": get_pareto_frontier(held_out_df, x_name="C", y_name="loss"),
        "C_split": float(held_out_df["C"].min()),
        "Cgrid": np.geomspace(df["C"].min(), df["C"].max(), 200),
    }


def hp_ground_truth(grid_df, form):
    """References for the optimal-HP panel: full-grid optima per scale, the held-out scale, a fit."""
    from core.fitting import build_fit

    target, scale_cols = form.target, form.input_cols
    scale_col = scale_cols[0]
    df = grid_df.dropna(subset=list(dict.fromkeys(scale_cols + [target, "loss"])))
    opt = df.loc[df.groupby(scale_cols, sort=False)["loss"].idxmin()]   # optimal HP per scale
    hi = opt[scale_col].max()
    return {
        "params": build_fit(opt, form_name=form.name, y_name=target)[0],
        "opt": opt,
        "test_opt": opt[opt[scale_col] == hi],
        "scale_grid": np.geomspace(opt[scale_col].min(), opt[scale_col].max(), 100),
        "scale_cols": scale_cols, "scale_col": scale_col, "target": target,
    }


# --- the two panel-0 variants (loss-vs-C for loss_law; optimal-HP-vs-scale for hp_law) ---

def _median_final_params(hists):
    return np.median([np.asarray(h["params"].iloc[-1], dtype=float) for h in hists], axis=0)


def _loss_panel0(ax, experiments, hists_by_label, form, refs):
    C_split = refs["C_split"]
    ax.axvspan(C_split, refs["test_known"]["C"].max(), color="#fca5a5", alpha=0.12, lw=0,
               label="held-out region")
    ax.scatter(refs["train_df"]["C"], refs["train_df"]["loss"], s=6, color="#d1d5db", alpha=0.5,
               zorder=1, label="all configs (per N,D scale)")
    ax.scatter(refs["front_obs"]["C"], refs["front_obs"]["loss"], s=16, color="#111827", zorder=3,
               label="compute-optimal (pool)")
    ax.scatter(refs["front_held"]["C"], refs["front_held"]["loss"], s=70, marker="X", color="#dc2626",
               zorder=5, edgecolor="white", linewidth=0.6, label="held-out eval frontier")
    Cgrid = refs["Cgrid"]
    seen = Cgrid <= C_split                                    # pool | held-out boundary
    ax.plot(Cgrid, form.to_L_vs_C(Cgrid, refs["params"]), "--", color="#111827", lw=2.0, zorder=4,
            label="full-grid fit")
    for label, spec in experiments.items():
        y = form.to_L_vs_C(Cgrid, _median_final_params(hists_by_label[label]))
        ax.plot(Cgrid[seen], y[seen], color=spec["color"], lw=1.8, label=label)      # train region
        ax.plot(Cgrid[~seen], y[~seen], color=spec["color"], lw=1.8, ls="--")        # held-out region
    ax.set_xscale("log"); ax.set_yscale("log")
    tidy_log_axis(ax, "x"); tidy_log_axis(ax, "y")
    ax.set_xlabel("compute C (FLOPs)"); ax.set_ylabel("loss L")
    ax.set_title("recovered law + eval regions"); ax.legend(fontsize=7)


def _hp_panel0(ax, experiments, hists_by_label, form, refs):
    scale_cols, scale_col, target = refs["scale_cols"], refs["scale_col"], refs["target"]
    opt, test_opt, grid = refs["opt"], refs["test_opt"], refs["scale_grid"]

    def X_of(g):  # single-input forms (bsz: D) pass 1-D; others hold extra inputs at the median
        return g if len(scale_cols) == 1 else np.column_stack(
            [g] + [np.full_like(g, opt[c].median()) for c in scale_cols[1:]])

    ax.scatter(opt[scale_col], opt[target], s=16, color="#111827", zorder=3, label="optimal per scale (full grid)")
    ax.scatter(test_opt[scale_col], test_opt[target], s=70, marker="X", color="#dc2626", zorder=5,
               edgecolor="white", linewidth=0.6, label=f"held-out ({scale_col} max)")
    ax.plot(grid, np.asarray(form.func(X_of(grid), refs["params"], return_log_loss=False), dtype=float),
            "--", color="#111827", lw=2.0, zorder=4, label="full-grid fit")
    for label, spec in experiments.items():
        mp = _median_final_params(hists_by_label[label])
        ax.plot(grid, np.asarray(form.func(X_of(grid), mp, return_log_loss=False), dtype=float),
                color=spec["color"], lw=1.8, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    tidy_log_axis(ax, "x"); tidy_log_axis(ax, "y")
    ax.set_xlabel(scale_col); ax.set_ylabel(f"optimal {target}")
    ax.set_title(f"recovered optimal {target} + held-out point"); ax.legend(fontsize=8)


# --- the canonical comparison figure ---

def _two_set_panel(ax, experiments, hists_by_label, ylab):
    """Fitted-law recovery MSE on both regions: pool (solid) vs held-out (dashed), median+IQR.

    Falls back to a single ``known_test_mse`` line for histories predating the two-set split.
    """
    cols = next(iter(hists_by_label.values()))[0].columns
    series = ([("law_mse_pool", "-", "o", "pool"), ("law_mse_held_out", "--", "s", "held-out")]
              if "law_mse_pool" in cols else [("known_test_mse", "-", "o", "held-out")])
    for label, spec in experiments.items():
        hists = hists_by_label[label]
        for key, ls, mk, tag in series:
            ys = [h[key].to_numpy() for h in hists]
            if all(np.isnan(y).all() for y in ys):       # e.g. no in-region set on a tiny grid
                continue
            x, med, q1, q3 = _band_series([h["total_compute_spent"] for h in hists], ys)
            ax.fill_between(x, q1, q3, color=spec["color"], alpha=0.12, lw=0)
            ax.plot(x, med, ls, marker=mk, ms=3, color=spec["color"], label=f"{label} ({tag})")
    ax.set_xscale("log"); ax.set_yscale("log")
    tidy_log_axis(ax, "x"); tidy_log_axis(ax, "y")
    ax.set_xlabel("compute spent (FLOPs)"); ax.set_ylabel(ylab)
    ax.set_title("law recovery: in-region vs held-out"); ax.legend(fontsize=7)


def comparison_figure(experiments, hists_by_label, form, refs, *, title, out_path, error_param=None,
                      domain="loss"):
    """Render the form-generic comparison figure to a high-DPI PNG.

    Panels: a domain-specific first panel (``domain="loss"`` -> recovered law + the eval regions via
    the compute frontier; ``domain="hp"`` -> recovered optimal HP vs scale), then the two-set law
    recovery MSE (in-region solid vs held-out dashed), the cumulative compute spent vs iteration, an
    optional ``|regret_<error_param>|`` headline panel, then one recovered-coefficient trajectory per
    interpretable parameter -- each a median line + IQR band per experiment, vs. the full-grid truth.
    """
    set_pub_style()
    interp_names = list(form.interpretable(refs["params"]).keys())
    gt_interp = form.interpretable(refs["params"])
    # skip the headline-error panel if this form doesn't log that regret column
    if error_param and f"regret_{error_param}" not in next(iter(hists_by_label.values()))[0].columns:
        error_param = None
    n_fixed = 3 + (1 if error_param else 0)   # panel0 + two-set recovery + compute-spent (+ error)
    ncols = 3
    nrows = ceil((n_fixed + len(interp_names)) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.5 * nrows))
    flat = np.atleast_2d(axes).flatten()
    fig.suptitle(title)

    # panel 0: domain-specific (loss-vs-C frontier, or optimal-HP-vs-scale)
    (_hp_panel0 if domain == "hp" else _loss_panel0)(flat[0], experiments, hists_by_label, form, refs)

    # trajectory panels: median + IQR band per experiment
    def traj(ax, valfn, title_, ylab, gt=None, logy=True, absval=False):
        for label, spec in experiments.items():
            hists = hists_by_label[label]
            x, med, q1, q3 = _band_series([h["total_compute_spent"] for h in hists],
                                          [valfn(h) for h in hists], absval)
            ax.fill_between(x, q1, q3, color=spec["color"], alpha=0.15, lw=0)
            ax.plot(x, med, "-o", color=spec["color"], ms=3, label=label)
        if gt is not None:
            ax.axhline(gt, ls="--", color="#111827", lw=1.3, label=f"full-grid={gt:.3g}")
        ax.set_xscale("log")
        tidy_log_axis(ax, "x")
        if logy:
            ax.set_yscale("log")
            tidy_log_axis(ax, "y")
        ax.set_xlabel("compute spent (FLOPs)"); ax.set_ylabel(ylab)
        ax.set_title(title_); ax.legend(fontsize=8)

    # two-set recovery panel: in-region (interpolation, solid) vs held-out (extrapolation, dashed)
    idx = 1
    mse_ylab = "optimal-HP log-MSE" if domain == "hp" else "loss MSE"
    _two_set_panel(flat[idx], experiments, hists_by_label, mse_ylab)
    idx += 1

    # cumulative compute spent vs iteration: how fast each method burns the acquisition budget
    ax = flat[idx]; idx += 1
    for label, spec in experiments.items():
        hists = hists_by_label[label]
        x, med, q1, q3 = _band_series([h["iter"].to_numpy() for h in hists],
                                      [h["total_compute_spent"].to_numpy() for h in hists])
        ax.fill_between(x, q1, q3, color=spec["color"], alpha=0.15, lw=0)
        ax.plot(x, med, "-o", color=spec["color"], ms=3, label=label)
    ax.set_yscale("log"); tidy_log_axis(ax, "y")
    ax.set_xlabel("iteration"); ax.set_ylabel("total compute spent (FLOPs)")
    ax.set_title("compute spent"); ax.legend(fontsize=8)

    if error_param:
        traj(flat[idx], lambda h: np.abs(h[f"regret_{error_param}"].to_numpy()),
             f"{error_param} error", f"|{error_param} regret|")
        idx += 1
    for name in interp_names:
        # value trajectory: map each iteration's optimizer-space params -> the interpretable value
        traj(flat[idx], lambda h, n=name: np.array([form.interpretable(p)[n] for p in h["params"]]),
             f"coefficient  {name}", name, gt=gt_interp[name], logy=(gt_interp[name] > 0))
        idx += 1

    for ax in flat[idx:]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print("saved", out_path)


def run_comparison(experiments, input_csv, seeds, *, out_path, title, task="loss_law", form_name=None,
                   niters=24, init_points=10, error_param="alpha"):
    """End-to-end comparison: clean the grid, run every experiment over seeds, render the figure.

    Ties ``clean_grid`` + ``run_experiments`` + the task's ground truth + ``comparison_figure`` so a
    per-experiment script is just an ``EXPERIMENTS`` dict + this one call. ``task`` picks the pipeline
    (``loss_law``/``hp_law``) and the panel-0 domain; ``form_name`` defaults per task. The shared
    ``form_name`` is injected into each experiment's driver flags.
    """
    from core.utils.data import clean_grid
    from core.forms import PARAMETRIC_FORMS

    form_name = form_name or ("L0" if task == "loss_law" else "steplaw_lr")
    grid = clean_grid(input_csv)
    for spec in experiments.values():
        spec.setdefault("form", form_name)
    hists = run_experiments(experiments, grid, seeds, out_dir=Path(out_path).parent, task=task,
                            niters=niters, init_points=init_points)
    form = PARAMETRIC_FORMS[form_name]
    if task == "hp_law":
        refs, domain = hp_ground_truth(pd.read_csv(grid), form), "hp"
    else:
        refs, domain = ground_truth(pd.read_csv(grid), form), "loss"
    comparison_figure(experiments, hists, form, refs, title=title, out_path=out_path,
                      error_param=error_param, domain=domain)

