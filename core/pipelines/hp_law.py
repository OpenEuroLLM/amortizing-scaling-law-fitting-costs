"""Optimal-HP scaling-law recovery (task ``hp_law``).

The step-law forms (steplaw_lr, steplaw_bsz) fit the **optimal hyperparameter** as a function of
scale (optimal lr vs (N,D); optimal batch size vs D), not loss. This runs the *same* surrogate
acquisition loop as ``loss_law`` (``_common.run_acquisition``) -- select a config, GP-infill the
grid's loss, iterate -- but swaps the extraction: instead of the loss Pareto frontier, each fit
reads the **optimal HP per scale** off the (observed + infilled) grid (the min-loss row at each
scale) and fits the step law to those optima.

The surrogate infill is what makes this more than a lookup: at scales where only a few HP settings
were observed, the GP-predicted loss fills the rest so the per-scale optimum can be estimated.
``--model_type none`` recovers the direct, model-free variant (fit on observed optima only).

Held-out score: the predicted-vs-actual optimal HP at the largest held-out scale (log-MSE). The
largest scale (largest N for lr, largest D for bsz) is held out and never acquired.

Dispatched from ``main.py`` (``python main.py hp_law ...``).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.forms import PARAMETRIC_FORMS
from core.fitting import build_fit, print_fit_results
from core.utils.plotting import set_pub_style
from core.pipelines import _common

# forms this task handles: those whose target is an HP (not loss)
HP_FORMS = [name for name, f in PARAMETRIC_FORMS.items() if f.target != "loss"]


def _reduce_to_optima(df, scale_cols):
    """Min-loss row per unique setting of ``scale_cols`` -> the empirical optimal HP at each scale."""
    return df.loc[df.groupby(scale_cols, sort=False)["loss"].idxmin()]


def add_args(parser):
    _common.add_acquisition_args(parser)
    parser.add_argument(
        '--form', type=str, default='steplaw_lr', choices=HP_FORMS,
        help="Optimal-HP form. 'steplaw_lr': optimal LR vs (N,D). 'steplaw_bsz': optimal batch "
        "size vs D. (Loss forms live under the 'loss_law' task.)"
    )


def run(args):
    # --- data + held-out split (by the form's scale axis) ---
    df, gp_features = _common.prepare_data(args)
    form = PARAMETRIC_FORMS[args.form]
    target = form.target                    # 'lr' or 'gbsz'
    scale_cols = form.input_cols            # [N, D] for lr; [D] for bsz
    scale_col = scale_cols[0]               # the axis whose largest value is held out

    df = df.dropna(subset=list(dict.fromkeys(scale_cols + [target, "loss"])))
    df["C"] = 6.0 * df["N"] * df["D"]       # compute, for the acquisition envelope

    # hold out the largest scale; the loop acquires only from the rest
    hi = df[scale_col].max()
    train_df = df[df[scale_col] < hi]
    test_df = df[df[scale_col] == hi]
    if test_df.empty or train_df[scale_col].nunique() < 3:
        raise ValueError(
            f"Need >= 3 train scales and a non-empty held-out set on '{scale_col}'; "
            f"got {train_df[scale_col].nunique()} train scales, {len(test_df)} held-out rows."
        )

    def _form_X(frame):
        # single-input forms (steplaw_bsz: D) want a 1-D array, matching build_fit's ravel
        X = frame[scale_cols].to_numpy(dtype=float)
        return X.ravel() if X.shape[1] == 1 else X

    # --- ground truth + held-out target + per-iteration callbacks ---
    # fit the step law on ALL data reduced to optima
    global_opt = _reduce_to_optima(pd.concat([train_df, test_df]), scale_cols)
    global_best_params, _ = build_fit(global_opt, form_name=args.form, y_name=target, fit_quality=args.fit_quality)
    print("Global optimal-HP fit (all data):")
    print_fit_results(form, global_best_params)

    # held-out optimal HP at the largest scale (reduce the held-out rows to their per-scale optima)
    test_opt = _reduce_to_optima(test_df, scale_cols)
    hp_true = test_opt[target].to_numpy(dtype=float)

    def fit_grid(df_pred):
        # read the optimal HP per scale off the (observed + infilled) grid, then fit the step law
        return build_fit(_reduce_to_optima(df_pred, scale_cols), form_name=args.form,
                         y_name=target, fit_quality=args.fit_quality)

    def heldout_metric(params):
        hp_pred = np.asarray(form.func(_form_X(test_opt), params, return_log_loss=False), dtype=float)
        return {"known_test_mse": float(np.mean((np.log(hp_pred) - np.log(hp_true)) ** 2))}

    # --- acquisition loop ---
    history, incumbents, interp_names, df_pred, acquired_idx = _common.run_acquisition(
        args, form, gp_features, train_df, global_best_params, fit_grid, heldout_metric
    )

    # --- report + save + plot ---
    print(f"{'-'*30}")
    print("Global optimal-HP fit (all data):")
    print_fit_results(form, global_best_params)
    print(f"Held-out ({scale_col}={hi:g}) optimal-{target} log-MSE: {history[-1]['known_test_mse']:.3e}")
    print(f"{'-'*30}")
    print("Run done!")

    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_parquet(output_dir / f"{args.filename}.parquet", index=False)
    print(f"Saved history to {output_dir / f'{args.filename}.parquet'}")

    if args.plot_figure:
        _plot(args, form, history_df, interp_names, df_pred, test_opt, hp_true,
              scale_cols, scale_col, target, global_best_params, output_dir)


def _plot(args, form, history_df, interp_names, df_pred, test_opt, hp_true,
          scale_cols, scale_col, target, global_best_params, output_dir):
    set_pub_style()
    n_extra = 2  # known-test log-MSE, optimal-HP-vs-scale
    total_panels = len(interp_names) + n_extra
    ncols = 3
    nrows = int(np.ceil(total_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    flat_axes = np.atleast_2d(axes).flatten()

    # per-param regret panels, then the held-out convergence panel
    panel_idx = _common.plot_regret_panels(flat_axes, 0, history_df, form, interp_names)
    ax = flat_axes[panel_idx]; panel_idx += 1
    ax.plot(history_df["iter"], history_df["known_test_mse"])
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_title(f"held-out optimal-{target} log-MSE")

    # optimal HP vs the primary scale axis: final-iteration optima, fitted curve, held-out point
    ax = flat_axes[panel_idx]; panel_idx += 1
    final_opt = _reduce_to_optima(df_pred, scale_cols)
    final_params = np.asarray(history_df.iloc[-1]["params"], dtype=float)
    ax.scatter(final_opt[scale_col], final_opt[target], s=16, color="#111827", label="optimal (acquired+infilled)")
    ax.scatter(test_opt[scale_col], hp_true, s=90, marker="X", color="#dc2626",
               edgecolor="white", linewidth=0.6, zorder=5, label=f"held-out ({scale_col} max)")
    grid = np.geomspace(final_opt[scale_col].min(), test_opt[scale_col].max(), 100)
    X = grid if len(scale_cols) == 1 else np.column_stack(
        [grid] + [np.full_like(grid, final_opt[c].median()) for c in scale_cols[1:]])
    ax.plot(grid, np.asarray(form.func(X, final_params, return_log_loss=False), dtype=float),
            color="#7c3aed", lw=2, label="step-law fit")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(scale_col); ax.set_ylabel(f"optimal {target}")
    ax.set_title(f"optimal {target} vs {scale_col}"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    _common.save_panels(fig, flat_axes, panel_idx, output_dir / f"{args.filename}.png")
