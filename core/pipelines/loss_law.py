"""Loss scaling-law recovery (task ``loss_law``).

Surrogate-based acquisition loop (see ``_common.run_acquisition``): each iteration selects a config,
(optionally) infills the unobserved grid's loss with a surrogate, fits a *loss* form, and logs
per-parameter regret and held-out MSE. Handles every form whose target is ``loss``:

- L-vs-C forms (L0, chinchilla3, chinchilla_interaction): held-out loss is scored via the form's
  compute marginalization (`to_L_vs_C`), and the run extrapolates loss beyond the observed compute.
- forms with no compute axis (width_depth): held-out loss is scored *directly* -- predict loss at
  the held-out config points via `form.func` -- and the C-extrapolation panels are skipped.

Dispatched from ``main.py`` (``python main.py loss_law ...``).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scaling.utils import get_pareto_frontier

from core import GRID_TARGET
from core.surrogates import build_surrogate
from core.forms import PARAMETRIC_FORMS
from core.fitting import build_fit, print_fit_results
from core.utils.data import data_preprocess
from core.utils.plotting import set_pub_style
from core.pipelines import _common

# forms this task handles: those that fit `loss` (steplaw_lr/bsz fit an HP -> hp_law task)
LOSS_FORMS = [name for name, f in PARAMETRIC_FORMS.items() if f.target == "loss"]


def _supports_compute_axis(form) -> bool:
    """True if the form marginalizes to L-vs-C (L0/chinchilla); False for width_depth."""
    try:
        form.to_L_vs_C(np.array([1.0]), np.asarray(form.initial_grid("debug")[0], dtype=float))
        return True
    except NotImplementedError:
        return False


def add_args(parser):
    _common.add_acquisition_args(parser)
    parser.add_argument(
        '--form', type=str, default='L0', choices=LOSS_FORMS,
        help="Loss form to fit. 'L0': L(C) = L0 + A*C^alpha. 'chinchilla3': L(N,D) = E + A/N^alpha "
        "+ B/D^beta. 'chinchilla_interaction': chinchilla3 plus an N*D interaction term. "
        "'width_depth': L(width,depth,D) = E + A/width^alpha + B/depth^beta + F/D^gamma -- no compute "
        "axis, so it is scored by direct held-out loss MSE (needs a --feature_set that exposes "
        "width/depth). Optimal-HP forms (steplaw_lr/bsz) live under the 'hp_law' task."
    )
    parser.add_argument(
        '--held_out_type', type=str, default='N', choices=['N', 'C'],
        help="Whether to hold out the largest N or the largest C for testing. Default is 'N'."
    )
    parser.add_argument(
        '--held_out_C_lower', type=float, default=0.5,
        help="held_out_C_lower*C_min defines the threshold for the held-out set when "
        "--held_out_type is 'C'. Default is 0.5."
    )


def run(args):
    # --- data + held-out split ---
    df, gp_features = _common.prepare_data(args)
    train_df, test_df_known, test_df_unknown = data_preprocess(
        df, held_out_type=args.held_out_type, held_out_C_lower=args.held_out_C_lower
    )

    form = PARAMETRIC_FORMS[args.form]
    supports_C = _supports_compute_axis(form)  # False for width_depth (no compute marginalization)

    # --- ground truth: global surrogate baseline + global parametric fit ---
    _global_df = pd.concat((train_df, test_df_known))
    print("\nBuilding global surrogate on all data as a baseline...")
    build_surrogate(
        np.log(_global_df[gp_features].to_numpy()),
        _global_df[GRID_TARGET].to_numpy()[:, None],
        np.log(test_df_known[gp_features].to_numpy()),
        model_type=args.model_type,
        use_mean_model=True if args.gp_mean_as_l0 else False,
        diagnose=args.diagnose_gp,
    )
    print("Fitting parametric form on the global surrogate...")
    global_best_params, _ = build_fit(
        _global_df, form_name=args.form, y_name="loss", fit_quality=args.fit_quality
    )
    print_fit_results(form, global_best_params)

    # --- held-out scoring setup + per-iteration callbacks ---
    # L-vs-C forms score against the held-out compute-optimal frontier + extrapolate beyond it;
    # width_depth scores loss directly at the held-out config points (no compute axis).
    if supports_C:
        _test_pareto = get_pareto_frontier(test_df_known, x_name="C", y_name="loss")
        C_known = _test_pareto["C"].to_numpy()
        L_known = _test_pareto["loss"].to_numpy()
        C_unknown = test_df_unknown["C"].to_numpy().astype(float)
    else:
        C_known = C_unknown = np.array([])
        L_known = test_df_known["loss"].to_numpy()
        X_known = test_df_known[form.input_cols].to_numpy()

    def fit_grid(df_pred):
        # build_fit reduces to the Pareto frontier internally when form.use_pareto
        return build_fit(df_pred, form_name=args.form, fit_quality=args.fit_quality)

    def heldout_metric(params):
        if supports_C:
            Lk = form.to_L_vs_C(C_known, params)
            Lu = form.to_L_vs_C(C_unknown, params)
            return {
                "known_test_mse": float(np.mean((Lk - L_known) ** 2)),
                "L_known": list(Lk),
                "C_unknown": list(C_unknown),
                "L_unknown": list(Lu),
            }
        Lk = np.asarray(form.func(X_known, params, return_log_loss=False))
        return {
            "known_test_mse": float(np.mean((Lk - L_known) ** 2)),
            "L_known": list(Lk),
            "C_unknown": [],
            "L_unknown": [],
        }

    # --- acquisition loop ---
    history, incumbents, interp_names, df_pred, acquired_idx = _common.run_acquisition(
        args, form, gp_features, train_df, global_best_params, fit_grid, heldout_metric
    )

    # --- report + save + plot ---
    print(f"{'-'*30}")
    print("Global surrogate fit results:")
    print_fit_results(form, global_best_params)
    print(f"Known Test MSE: {history[-1]['known_test_mse']:.2e}")
    if supports_C:
        print("Unknown Test Predictions:")
        for i in range(len(C_unknown)):
            print(f"    C={C_unknown[i]:.2e} -> L={history[-1]['L_unknown'][i]:.3f}")
    print(f"{'-'*30}")
    print("Run done!")

    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_parquet(output_dir / f"{args.filename}.parquet", index=False)
    print(f"Saved history to {output_dir / f'{args.filename}.parquet'}")

    if args.plot_figure:
        _plot(args, form, supports_C, history_df, interp_names, train_df, acquired_idx,
              df_pred, C_known, L_known, C_unknown, global_best_params, output_dir)


def _plot(args, form, supports_C, history_df, interp_names, train_df, acquired_idx,
          df_pred, C_known, L_known, C_unknown, global_best_params, output_dir):
    # panels: one regret plot per interpretable param, plus known-test mse, compute/iter, and
    # (for L-vs-C forms) the fit-comparison plot. Laid out dynamically.
    set_pub_style()
    n_extra_panels = 3  # known-test mse, compute/iter, fit-comparison (last skipped w/o compute axis)
    total_panels = len(interp_names) + n_extra_panels
    ncols = 3
    nrows = int(np.ceil(total_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    flat_axes = np.atleast_2d(axes).flatten()

    # per-param regret panels, then the two trajectory panels
    panel_idx = _common.plot_regret_panels(flat_axes, 0, history_df, form, interp_names)
    for col, title, yscale in [
        ("known_test_mse", "known-test MSE", "log"),
        ("iter_compute", "compute / iter", "log"),
    ]:
        ax = flat_axes[panel_idx]
        ax.plot(history_df["iter"], history_df[col])
        if yscale:
            ax.set_yscale(yscale)
        ax.set_xlabel("iteration")
        ax.set_title(title)
        panel_idx += 1

    # fit-comparison panel: real vs fantasized parametric fits as L vs C (only for L-vs-C forms)
    ax = flat_axes[panel_idx]
    panel_idx += 1
    if supports_C:
        obs_df = train_df.loc[acquired_idx]
        real_params, _ = build_fit(obs_df, form_name=args.form, fit_quality=args.fit_quality)
        fant_params = history_df.iloc[-1]["params"]
        _obs_pareto = get_pareto_frontier(obs_df, x_name="C", y_name="loss")
        ax.scatter(_obs_pareto["C"], _obs_pareto["loss"], marker="x", color="black", zorder=3, label="observed")
        ax.scatter(C_known, L_known, marker="x", color="tab:blue", zorder=3, label="held out")
        C_grid = np.geomspace(obs_df["C"].min(), C_unknown.max(), 200)
        seen = C_grid <= obs_df["C"].max()
        for params, color, label in [(real_params, "tab:green", "subsampled grid"), (fant_params, "red", "fantasized dense grid")]:
            L_hat = form.to_L_vs_C(C_grid, np.asarray(params))
            ax.plot(C_grid[seen], L_hat[seen], color=color, ls="-", label=label)
            ax.plot(C_grid[~seen], L_hat[~seen], color=color, ls="--")
        L_global_hat = form.to_L_vs_C(C_grid, np.asarray(global_best_params))
        ax.plot(C_grid[seen], L_global_hat[seen], color="tab:purple", ls="-.", label="all data fit")
        ax.plot(C_grid[~seen], L_global_hat[~seen], color="tab:purple", ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("C"); ax.set_ylabel("loss")
        ax.legend(fontsize=15)
        ax.set_title("real vs fantasized fit (dashed = beyond seen C)")
    else:
        ax.axis("off")  # width_depth: no compute axis to plot against

    _common.save_panels(fig, flat_axes, panel_idx, output_dir / f"{args.filename}.png")
