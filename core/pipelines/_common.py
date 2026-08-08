"""Shared machinery for the acquisition-loop tasks (loss_law, hp_law).

Both tasks run the same loop -- select a config, GP-infill the unobserved grid's loss, fit a form,
log regret/MSE -- and differ only in two callbacks:

- ``fit_grid(df_pred) -> (params, fit_loss)``: extract the task's relevant subset from the
  (observed + infilled) grid and fit the form. loss_law fits the loss form (Pareto handled by
  build_fit); hp_law reduces to the optimal HP per scale first.
- ``heldout_metric(params) -> dict``: task-specific held-out fields merged into each history row
  (e.g. loss_law's known_test_mse + L-vs-C curves; hp_law's optimal-HP MSE).
"""
import warnings

import numpy as np
import pandas as pd

from scaling.utils import get_pareto_frontier

from core import GRID_TARGET, MAX_ITERS
from core.selection import SELECTION
from core.surrogates import MODELS, build_surrogate
from core.fitting import build_fit, print_fit_results  # noqa: F401  (re-exported for pipelines)
from core.utils.data import next_batch_compute, resolve_grid_features, FEATURE_SETS


def score_law_frontier(df_set, form, params, supports_C):
    """MSE of the fitted law's predicted loss vs. actual on one eval set.

    L-vs-C forms are scored on the set's compute-optimal frontier (predicted ``L(C)`` vs. the
    frontier loss); forms with no compute axis (width_depth) are scored directly at the set's config
    points. Returns ``nan`` for an empty set (e.g. a grid too small to carve an in-region eval set).
    """
    if df_set is None or len(df_set) == 0:
        return float("nan")
    if supports_C:
        front = get_pareto_frontier(df_set, x_name="C", y_name="loss")
        L_hat = form.to_L_vs_C(front["C"].to_numpy(dtype=float), params)
        L_true = front["loss"].to_numpy(dtype=float)
    else:
        L_hat = np.asarray(form.func(df_set[form.input_cols].to_numpy(), params, return_log_loss=False))
        L_true = df_set["loss"].to_numpy(dtype=float)
    return float(np.mean((L_hat - L_true) ** 2))


def add_acquisition_args(parser):
    """CLI flags shared by every acquisition-loop task. Tasks add their own ``--form``."""
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--niters', type=int, default=None)
    parser.add_argument('--C_budget', type=float, default=None)
    parser.add_argument('--init_points', type=int, default=25)
    parser.add_argument(
        '--init_compute_factor', type=float, default=10.0,
        help="Factor to multiply the minimum compute cost to set the initial envelope for sampling."
    )
    parser.add_argument(
        '--C_window', type=float, nargs=2, default=[1e-3, 1e3], metavar=('LO', 'HI'),
        help="Factors on the max. seen compute to determine permissible range for sampling."
    )
    parser.add_argument(
        '--model_type', type=str, default='gp', choices=list(MODELS),
        help="Surrogate used to infill the unobserved grid before the fit. 'none' is model-free: "
        "fit directly on the observed points, no infill (turns --selection random into a fully "
        "model-free baseline)."
    )
    parser.add_argument('--selection', type=str, default='random', choices=list(SELECTION))
    parser.add_argument(
        '--feature_set', type=str, default='default', choices=list(FEATURE_SETS),
        help="Which features the GP conditions on. 'default': N, D, lr, gbsz. 'arch': + the "
        "architectural HPs (width, depth, ffn, num_heads). 'data': + the data-axis HPs "
        "(unique_tokens, seq_len). 'full': + both. Auto-narrowed to columns present and "
        "non-constant, so absent/all-null/fixed knobs drop out (extra modes collapse to default)."
    )
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        '--fit_quality', type=str, default='debug', choices=['debug', 'dense', 'super_dense']
    )
    parser.add_argument('--filename', default="history")
    parser.add_argument('--plot_figure', action='store_true')
    parser.add_argument(
        '--gp_mean_as_l0', action='store_true',
        help="If set, the GP mean function is initialized with the previous iteration's best-fit L0. "
        "If not set, the GP mean is a constant function (default)."
    )
    parser.add_argument(
        '--diagnose_gp', action='store_true',
        help="If set, prints GP diagnostics (MLL, train MSE, noise, lengthscales, mean parameters) after fitting."
    )


def prepare_data(args):
    """Read the CSV, resolve the GP feature set for this dataset, and drop NaN-poisoned rows.

    Returns ``(df, gp_features)``. Tasks then apply their own held-out split (loss_law by N/C,
    hp_law by the form's scale axis).
    """
    df = pd.read_csv(args.input)
    gp_features = resolve_grid_features(df, args.feature_set)
    if not gp_features:
        raise ValueError(f"No usable GP features resolved from {args.input} for --feature_set {args.feature_set}.")
    df = df.dropna(subset=gp_features + [GRID_TARGET])
    print(f"GP features ({args.feature_set}): {gp_features}")
    return df, gp_features


def run_acquisition(args, form, gp_features, train_df, global_best_params, fit_grid, heldout_metric):
    """The shared acquisition loop.

    ``fit_grid(df_pred)`` extracts + fits (task-specific); ``heldout_metric(params)`` returns extra
    per-iteration history fields (task-specific). Returns ``(history, incumbents, interp_names,
    df_pred, acquired_idx)`` -- the last two for the caller's plotting.
    """
    if args.niters is None and args.C_budget is None:
        raise ValueError("Either --niters or --C_budget must be specified.")
    if args.niters is not None and args.C_budget is not None:
        raise ValueError("Only one of --niters or --C_budget can be specified.")
    n_iters = args.init_points + (args.niters or (MAX_ITERS - args.init_points))

    splitter = SELECTION[args.selection](seed=args.seed)
    if args.model_type == "none" and getattr(splitter, "needs_surrogate", False):
        raise ValueError(
            f"--model_type none (model-free) is incompatible with surrogate-based selection "
            f"'{args.selection}' (it needs the GP to pick points). Use a surrogate-free selector, "
            f"e.g. --selection random."
        )
    rng = np.random.RandomState(args.seed)

    # initialization: seed with the cheapest point, then fill up to init_points under the envelope
    C_lim = train_df.C.min() * args.init_compute_factor
    pool = next_batch_compute(train_df, acquired_idx=None, C_max=C_lim, C_window=args.C_window)
    acquired_idx = pd.Index(rng.choice(pool.index, args.init_points, replace=False))
    total_compute_spent = train_df.loc[acquired_idx, "C"].sum()

    history = []
    df_pred = None
    interp_names = list(form.interpretable(global_best_params).keys())
    incumbents = {name: np.inf for name in interp_names}
    for iter_idx in range(args.init_points, n_iters + 1):
        # budget checks
        if args.C_budget is not None and total_compute_spent >= args.C_budget:
            print(f"\n\n{'=' * 50}\nReached total compute budget of {args.C_budget}. Stopping iterations.\n{'=' * 50}\n")
            break
        if len(acquired_idx) == len(train_df):
            print(f"\n\n{'=' * 50}\nAll {len(train_df)} points acquired. Stopping iterations.\n{'=' * 50}\n")
            break

        # sample a new point under the cost envelope; its C is this iteration's cost
        pool = next_batch_compute(train_df, acquired_idx, C_max=C_lim, C_window=args.C_window)
        # surrogate-guided selectors (e.g. max_variance) need a GP fit on the current
        # observations *before* picking; fit it once and hand it to next_point.
        # farthest_point / random ignore `surrogate` and use `acquired_idx` instead.
        sel_surrogate = None
        if getattr(splitter, "needs_surrogate", False) and len(acquired_idx) >= 2:
            _obs = train_df.loc[acquired_idx]
            try:
                sel_surrogate = MODELS[args.model_type]()
                sel_surrogate.fit(np.log(_obs[gp_features].to_numpy()), _obs[GRID_TARGET].to_numpy()[:, None])
            except Exception as e:
                # GP fit can raise NotPSDError: the kernel matrix is ill-conditioned (no inverse --
                # a kernel must be positive-definite), which happens with few / near-duplicate points.
                # We fall back to a random pick for this one iteration; warn so that's not silent.
                warnings.warn(
                    f"[iter {iter_idx - args.init_points}] selection GP fit failed "
                    f"({type(e).__name__}: {e}); using a RANDOM pick this iteration "
                    f"instead of '{args.selection}'.",
                    RuntimeWarning, stacklevel=2,
                )
                sel_surrogate = None
        new_idx = splitter.next_point(train_df, pool.index,
                                      acquired_idx=acquired_idx, surrogate=sel_surrogate)
        acquired_idx = acquired_idx.append(pd.Index([new_idx]))
        iter_compute = train_df.loc[new_idx, "C"]
        total_compute_spent += iter_compute

        observations = train_df.loc[acquired_idx]
        remaining_idx = train_df.index.difference(acquired_idx)

        # fit the surrogate on observations (log-features), predict the unacquired rest.
        # model-free (NullSurrogate) predicts NaN everywhere, so the dropna below leaves only
        # the observed points -- a fit-on-observations baseline, with no branch here.
        pred_y, heldout_mse = build_surrogate(
            np.log(observations[gp_features].to_numpy()),
            observations[GRID_TARGET].to_numpy()[:, None],
            np.log(train_df.loc[remaining_idx, gp_features].to_numpy()),
            test_y=train_df.loc[remaining_idx, GRID_TARGET].to_numpy(),
            model_type=args.model_type,
            use_mean_model=True if args.gp_mean_as_l0 else False,
            diagnose=args.diagnose_gp
        )
        # true loss where observed, predicted loss elsewhere, then fit the mixed data.
        # NaN predictions (model-free) drop out, leaving the observed-only fit.
        df_pred = train_df.copy()
        df_pred.loc[remaining_idx, GRID_TARGET] = pred_y
        df_pred = df_pred.dropna(subset=[GRID_TARGET])
        best_params, best_loss = fit_grid(df_pred)
        print(f"#Iteration {iter_idx}: Fitting loss={best_loss:2.2e}", end="; ")
        print_fit_results(form, best_params)

        # regrets in interpretable space; irreducible-loss-like terms (form.signed_params) keep sign
        interp = form.interpretable(best_params)
        global_interp = form.interpretable(global_best_params)
        regrets = {
            name: (interp[name] - global_interp[name]) if name in form.signed_params
            else abs(interp[name] - global_interp[name])
            for name in interp_names
        }

        history.append({
            "iter": iter_idx,
            "n_acquired": len(acquired_idx),
            "acquired_idx": int(new_idx),
            "iter_compute": float(iter_compute),
            "total_compute_spent": float(total_compute_spent),
            "heldout_mse": float(heldout_mse),
            "heldout_size": len(remaining_idx),
            **heldout_metric(best_params),
            "params": list(best_params),
            "fit_loss": float(best_loss),
            **{f"regret_{name}": float(regrets[name]) for name in interp_names},
        })
        # updating incumbent tracker
        # signed params: incumbent is the value closest to zero, sign preserved; others: plain min
        for name in interp_names:
            if name in form.signed_params:
                incumbents[name] = min(incumbents[name], history[-1][f"regret_{name}"], key=abs)
            else:
                incumbents[name] = min(incumbents[name], history[-1][f"regret_{name}"])

        # update C_lim envelope for sampling: ceiling stays one growth step above the max cost seen
        C_lim = observations.C.max()

    print(f"{'='*30}\nIncumbent regrets:")
    for name in interp_names:
        tag = " (signed)" if name in form.signed_params else ""
        print(f"{name}{tag}: {incumbents[name]:.4f}")
    print(f"{'='*30}")
    return history, incumbents, interp_names, df_pred, acquired_idx


# --- shared plotting helpers (both tasks lay out a grid of per-param regret panels) ---

def plot_regret_panels(flat_axes, start_idx, history_df, form, interp_names):
    """Draw one incumbent-regret panel per interpretable param; return the next free panel index.

    Signed params (irreducible-loss-like) track the running value closest to zero with sign
    preserved (matches the printed incumbents); others track the running minimum on a log axis.
    """
    idx = start_idx
    for name in interp_names:
        ax = flat_axes[idx]
        col = f"regret_{name}"
        if name in form.signed_params:
            vals = history_df[col].to_numpy()
            signed_running, best = [], vals[0]
            for v in vals:
                if abs(v) < abs(best):
                    best = v
                signed_running.append(best)
            ax.plot(history_df["iter"], signed_running)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_title(f"incumbent regret {name} (signed)")
        else:
            ax.plot(history_df["iter"], np.minimum.accumulate(history_df[col]))
            ax.set_yscale("log")
            ax.set_title(f"incumbent regret {name}")
        ax.set_xlabel("iteration")
        idx += 1
    return idx


def save_panels(fig, flat_axes, panel_idx, path):
    """Blank any unused trailing axes, tidy, and write the figure."""
    for ax in flat_axes[panel_idx:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved figure to {path}")
