"""Loss scaling-law recovery (task ``loss_law``).

Surrogate-based acquisition loop (see ``_common.run_acquisition``): each iteration selects a config,
(optionally) infills the unobserved grid's loss with a surrogate, fits a *loss* form, and logs
per-parameter regret plus the fitted law's MSE on two eval sets.

The grid is split (``split_search_space``) by a fraction along ``--split_axis`` into two parts: the
acquire-able **pool** (everything below the threshold -- the loop draws from it) and the top-fraction
**held-out** region (extrapolation, never acquired). Each iteration scores the fitted law on both ->
``law_mse_pool`` (in-region recovery) and ``law_mse_held_out`` (extrapolation); their gap is the
in-region->held-out generalization gap. Handles every form whose target is ``loss``:

- L-vs-C forms (L0, chinchilla3, chinchilla_interaction): each region is scored on its compute-optimal
  frontier via `to_L_vs_C`, and the run extrapolates loss beyond the observed compute.
- forms with no compute axis (width_depth): loss is scored *directly* at the config points via
  `form.func`, and the C-extrapolation panels are skipped.

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
from core.utils.data import split_search_space
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
        '--split_axis', type=str, default='C', choices=['C', 'N'],
        help="Axis the fraction split runs along. 'C': compute (6ND). 'N': model size. The top "
        "fraction by this axis is the held-out (extrapolation) region. Default 'C'."
    )
    parser.add_argument(
        '--held_out_frac', type=float, default=0.2,
        help="Fraction of the grid held out along --split_axis (largest scales) for the "
        "extrapolation eval set. Everything below the threshold is the acquire-able pool, on which "
        "the in-region (interpolation) recovery is scored. Default 0.2."
    )


def run(args):
    # --- data + two-way fraction split (acquire-able pool / held-out) ---
    df, gp_features = _common.prepare_data(args)
    pool_df, held_out_df = split_search_space(
        df, held_out_frac=args.held_out_frac, axis=args.split_axis
    )
    train_df = pool_df                        # the loop acquires only from the pool
    test_df_known = held_out_df
    print(f"Split (axis={args.split_axis}): pool={len(pool_df)} / held-out={len(held_out_df)}")

    form = PARAMETRIC_FORMS[args.form]
    supports_C = _supports_compute_axis(form)  # False for width_depth (no compute marginalization)

    # --- ground truth: global surrogate baseline + global parametric fit (all data) ---
    _global_df = pd.concat((pool_df, held_out_df))
    print("\nBuilding global surrogate on all data as a baseline...")
    global_pred, global_mse, gp_surrogate = build_surrogate(
        np.log(_global_df[gp_features].to_numpy()),
        _global_df[GRID_TARGET].to_numpy()[:, None],
        np.log(held_out_df[gp_features].to_numpy()),
        model_type=args.model_type,
        mean_form=...,
        # use_mean_model=True if args.gp_mean_as_l0 else False,
        kernel=...,
        diagnose=args.diagnose_gp,
    )
    print(f"Fitting parametric form on all of the data {_global_df.shape}...")
    global_best_params, _ = build_fit(
        _global_df, form_name=args.form, y_name="loss", fit_quality=args.fit_quality
    )
    print_fit_results(form, global_best_params)
    
    ### get scaling law parameters from training data all methods should converge to this
    print(f"Fitting parametric form on all of the training data {train_df.shape}...")

    # build fit first extracts pareto frontier and then fits the L0 parametric form
    train_best_params, train_best_loss = build_fit(
        train_df, form_name=args.form, y_name="loss", fit_quality=args.fit_quality
    )
    print_fit_results(form, train_best_params)
    
    # Compute train and held-out pareto frontiers
    combined = pd.concat([train_df, test_df_known])
    global_pareto = get_pareto_frontier(combined, x_name='C', y_name="loss")
    _pool_pareto = global_pareto.loc[global_pareto.index.isin(train_df.index)]
    _held_pareto  = global_pareto.loc[global_pareto.index.isin(test_df_known.index)]

    # --- scoring setup + per-iteration callbacks ---
    # Score the fitted law on both regions: the acquire-able pool (in-region recovery) and the
    # held-out region (extrapolation). L-vs-C forms score on each region's compute-optimal frontier;
    # width_depth scores loss directly at the config points. The held-out frontier + a synthetic
    # beyond-grid extension also drive the fit-comparison plot panel.
    if supports_C:
        C_known = _held_pareto["C"].to_numpy()
        L_known = _held_pareto["loss"].to_numpy()
        C_unknown = C_known.max() * 10.0 ** np.arange(1, 6)   # extrapolation curve beyond the grid
    else:
        C_known = C_unknown = np.array([])

    def fit_grid(df_pred):
        # build_fit reduces to the Pareto frontier internally when form.use_pareto
        return build_fit(df_pred, form_name=args.form, fit_quality=args.fit_quality)

    def heldout_metric(params):
        # mse_pool = _common.score_law_frontier(pool_df, form, params, supports_C)
        # mse_held = _common.score_law_frontier(held_out_df, form, params, supports_C)
        mse_pool = _common.mse_on_frontier(_pool_pareto, form, params)
        mse_held = _common.mse_on_frontier(_held_pareto, form, params)
        
        out = {
            "law_mse_pool": mse_pool,
            "law_mse_held_out": mse_held,
            "known_test_mse": mse_held,          # back-compat alias (held-out score)
        }
        if supports_C:
            out.update(L_known=list(form.to_L_vs_C(C_known, params)),
                       C_unknown=list(C_unknown),
                       L_unknown=list(form.to_L_vs_C(C_unknown, params)))
        else:
            out.update(L_known=[], C_unknown=[], L_unknown=[])
        return out

    # --- acquisition loop ---
    history, incumbents, interp_names, df_pred, acquired_idx = _common.run_acquisition(
        args, form, gp_features, train_df, global_best_params, fit_grid, heldout_metric
    )

    # --- report + save + plot ---
    print(f"{'-'*30}")
    print("Global surrogate fit results:")
    print_fit_results(form, global_best_params)
    print(f"Pool (in-region) MSE: {history[-1]['law_mse_pool']:.2e}  |  "
          f"Held-out eval MSE: {history[-1]['law_mse_held_out']:.2e}")
    if supports_C:
        print("Extrapolation beyond the grid:")
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
              held_out_df, C_known, L_known, C_unknown, global_best_params, output_dir)


def _plot(args, form, supports_C, history_df, interp_names, train_df, acquired_idx,
          held_out_df, C_known, L_known, C_unknown, global_best_params, output_dir):
    # panels: one regret plot per interpretable param, plus the two-set recovery curve, compute/iter,
    # and (for L-vs-C forms) the region-shaded fit-comparison plot. Laid out dynamically.
    set_pub_style()
    n_extra_panels = 3  # two-set MSE, compute/iter, fit-comparison (last skipped w/o compute axis)
    total_panels = len(interp_names) + n_extra_panels
    ncols = 3
    nrows = int(np.ceil(total_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    flat_axes = np.atleast_2d(axes).flatten()

    # per-param regret panels, then the two-set recovery panel + compute/iter
    panel_idx = _common.plot_regret_panels(flat_axes, 0, history_df, form, interp_names)

    # two-set recovery: pool (in-region, solid) vs held-out (extrapolation, dashed)
    ax = flat_axes[panel_idx]; panel_idx += 1
    ax.plot(history_df["iter"], history_df["law_mse_pool"], "-o", ms=3, color="#0072B2", label="pool (in-region)")
    ax.plot(history_df["iter"], history_df["law_mse_held_out"], "--s", ms=3, color="#D55E00", label="held-out")
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_title("law recovery MSE"); ax.legend(fontsize=9)

    ax = flat_axes[panel_idx]; panel_idx += 1
    ax.plot(history_df["iter"], history_df["iter_compute"])
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_title("compute / iter")

    # fit-comparison panel: real vs fantasized parametric fits as L vs C, with the train/held-out
    # regions shaded (only for L-vs-C forms)
    ax = flat_axes[panel_idx]
    panel_idx += 1
    if supports_C:
        obs_df = train_df.loc[acquired_idx]
        real_params, _ = build_fit(obs_df, form_name=args.form, fit_quality=args.fit_quality)
        _obs_pareto = get_pareto_frontier(obs_df, x_name="C", y_name="loss")
        C_split = float(held_out_df["C"].min())          # pool | held-out boundary
        ax.axvspan(C_split, float(held_out_df["C"].max()), color="#fca5a5", alpha=0.12, lw=0,
                   label="held-out region")
        ax.scatter(_obs_pareto["C"], _obs_pareto["loss"], marker="x", color="black", zorder=3, label="observed")
        ax.scatter(C_known, L_known, marker="X", color="#dc2626", zorder=5, label="held-out eval")
        C_grid = np.geomspace(obs_df["C"].min(), C_unknown.max(), 200)
        seen = C_grid <= obs_df["C"].max()
        # fit on observed points only; add the surrogate-infilled ("fantasized") fit only when a
        # surrogate ran -- model-free imputes nothing, so it would duplicate the observed-only curve.
        curves = [(real_params, "tab:green", "subsampled grid")]
        if args.model_type != "none":
            curves.append((history_df.iloc[-1]["params"], "red", "fantasized dense grid"))
        for params, color, label in curves:
            L_hat = form.to_L_vs_C(C_grid, np.asarray(params))
            ax.plot(C_grid[seen], L_hat[seen], color=color, ls="-", label=label)
            ax.plot(C_grid[~seen], L_hat[~seen], color=color, ls="--")
        L_global_hat = form.to_L_vs_C(C_grid, np.asarray(global_best_params))
        ax.plot(C_grid[seen], L_global_hat[seen], color="tab:purple", ls="-.", label="all data fit")
        ax.plot(C_grid[~seen], L_global_hat[~seen], color="tab:purple", ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("C"); ax.set_ylabel("loss")
        ax.legend(fontsize=12)
        title = "fit (dashed = beyond seen C)" if args.model_type == "none" else \
                "real vs fantasized fit (dashed = beyond seen C)"
        ax.set_title(title)
    else:
        ax.axis("off")  # width_depth: no compute axis to plot against

    _common.save_panels(fig, flat_axes, panel_idx, output_dir / f"{args.filename}.png")
