"""Shared machinery for the acquisition-loop tasks (loss_law, hp_law).

Both tasks run the same loop -- select a config, GP-infill the unobserved grid's loss, fit a form,
log regret/MSE -- and differ only in two callbacks:

- ``fit_grid(df_pred) -> (params, fit_loss)``: extract the task's relevant subset from the
  (observed + infilled) grid and fit the form. loss_law fits the loss form (Pareto handled by
  build_fit); hp_law reduces to the optimal HP per scale first.
- ``heldout_metric(params) -> dict``: task-specific held-out fields merged into each history row
  (e.g. loss_law's known_test_mse + L-vs-C curves; hp_law's optimal-HP MSE).
"""
import json
import sys
import warnings
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scaling.utils import get_pareto_frontier

from core import GRID_TARGET, MAX_ITERS
from core.selection import SELECTION
from core.surrogates import MODELS, build_surrogate
from core.surrogates.gp import KERNEL_NAMES, MEAN_FORM_NAMES
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


def mse_on_frontier(front: pd.DataFrame, form, params) -> float:
    if front.empty:
        return float("nan")
    L_hat = form.to_L_vs_C(front["C"].to_numpy(dtype=float), params)
    L_true = front["loss"].to_numpy(dtype=float)
    return float(np.mean((L_hat - L_true) ** 2))


def seed_everything(seed: int) -> None:
    """Seed every RNG the acquisition loop touches, for run-to-run reproducibility.

    ``args.seed`` already drives the NumPy point-selection RNGs (each selector owns a
    ``RandomState``); this additionally pins the *global* NumPy RNG and torch's global RNG.
    The latter is what botorch's ``fit_gpytorch_mll`` draws on for its multi-start L-BFGS
    restart initialization -- without it the GP fit (and every GP-guided selector) is
    non-deterministic even at a fixed ``--seed``. torch is imported lazily so the model-free
    and non-GP paths don't pay for it.
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


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
        '--init_practice', action='store_true',
        help="Practice-style init: at the smallest N, form a pool from the full HP grid over the "
        "k smallest D values (--init_practice_ds), then randomly sample --init_points rows from "
        "that pool (seed-dependent). Sets C_lim to max C among sampled rows; init compute is the "
        "sum of C over sampled rows only. Default init samples randomly under the envelope."
    )
    parser.add_argument(
        '--init_practice_ds', type=int, default=4,
        help="With --init_practice: how many smallest D values at min N define the init pool "
        "(typical range 1–4). Ignored unless --init_practice is set."
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
        '--lcb_kappa', type=float, default=2.0,
        help="LCB exploration weight (mu - kappa*sigma). Only used by --selection lcb. "
        "Typical 1-3; larger => more exploratory."
    )
    parser.add_argument(
        '--feature_set', type=str, default='default', choices=list(FEATURE_SETS),
        help="Which features the GP/Model conditions on. 'default': N, D, lr, gbsz. 'arch': + the "
        "architectural HPs (width, depth, ffn, num_heads). 'data': + the data-axis HPs "
        "(unique_tokens, seq_len). 'full': + both. Auto-narrowed to columns present and "
        "non-constant, so absent/all-null/fixed knobs drop out (extra modes collapse to default)."
    )
    parser.add_argument(
        "--hp_list", type=str, nargs="+", default=None,
        help="Extra HP columns to be union-ed of the `feature_set` for the model."
    )
    parser.add_argument(
        '--y_target', type=str, default=GRID_TARGET,
        help=f"Column the model learns, or the dependent variable. Default '{GRID_TARGET}'."
    )

    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        '--fit_quality', type=str, default='debug', choices=['debug', 'dense', 'super_dense']
    )
    parser.add_argument('--filename', default="history")
    parser.add_argument('--plot_figure', action='store_true')
    parser.add_argument(
        '--gp_kernel', type=str, default='matern', choices=list(KERNEL_NAMES),
        help="GP covariance kernel: 'matern' (Matern-5/2 with ARD, default), 'linear', "
        "'linear+matern', or 'linear_nd+matern_hp' (linear on N,D + Matern on the HP dims)."
    )
    parser.add_argument(
        '--gp_mean_form', type=str, default=None, choices=list(MEAN_FORM_NAMES),
        help="Parametric form used as the GP mean, fitted during MLL optimization "
        "(keys mirror --form). If unset, the GP mean is a constant function (default)."
    )
    parser.add_argument(
        '--gp_mean_warmstart', action='store_true',
        help="Initialize the GP mean with the best parametric fit seen so far "
        "(lowest in-data fit loss). Only relevant when --gp_mean_form is set."
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
    gp_features = resolve_grid_features(df, args.feature_set, args.hp_list)
    if not gp_features:
        raise ValueError(f"No usable GP features resolved from {args.input} for --feature_set {args.feature_set}.")
    df = df.dropna(subset=gp_features + [args.y_target])
    print(f"GP features ({args.feature_set}): {gp_features}")
    return df, gp_features


def _init_acquired_idx_default(train_df, init_points, rng, C_lim, C_window):
    """Random init: sample ``init_points`` rows from the pool under the compute envelope."""
    pool = next_batch_compute(train_df, acquired_idx=None, C_max=C_lim, C_window=C_window)
    if len(pool) < init_points:
        raise ValueError(
            f"Init pool has {len(pool)} rows but --init_points={init_points}; "
            "widen --C_window or lower --init_points."
        )
    return pd.Index(rng.choice(pool.index, init_points, replace=False))


def _init_acquired_idx_practice(train_df, init_points, init_practice_ds, y_target, rng):
    """Practice-style init: full HP grid at smallest N for ``init_practice_ds`` smallest Ds.

    Randomly samples ``init_points`` rows from that pool (seed-dependent). Returns
    ``(acquired_idx, practice_pool_idx)``.
    """
    needed = {"N", "D", y_target, "C"}
    missing = needed - set(train_df.columns)
    if missing:
        raise ValueError(f"--init_practice requires columns {sorted(needed)} on the acquire pool; missing {sorted(missing)}.")
    if init_practice_ds < 1:
        raise ValueError(f"--init_practice_ds must be >= 1, got {init_practice_ds}.")
    min_N = train_df["N"].min()
    candidates = train_df[train_df["N"] == min_N]
    if candidates.empty:
        raise ValueError(f"--init_practice: no rows at min N={min_N}.")
    k = min(init_practice_ds, candidates["D"].nunique())
    smallest_Ds = np.sort(candidates["D"].unique())[:k]
    practice_pool = candidates[candidates["D"].isin(smallest_Ds)]
    if practice_pool.empty:
        raise ValueError("--init_practice: found no initial pool rows.")
    if len(practice_pool) < init_points:
        raise ValueError(
            f"--init_practice pool has {len(practice_pool)} rows at min N over {k} D value(s) "
            f"but --init_points={init_points}; increase --init_practice_ds or lower --init_points."
        )
    acquired_idx = pd.Index(rng.choice(practice_pool.index, init_points, replace=False))
    return acquired_idx, practice_pool.index


def _param_regrets(interp, ref_interp, form, interp_names) -> dict[str, float]:
    """Interpretable-parameter regret vs a reference fit (signed for ``form.signed_params``)."""
    return {
        name: (interp[name] - ref_interp[name]) if name in form.signed_params
        else abs(interp[name] - ref_interp[name])
        for name in interp_names
    }


def _update_incumbents(incumbents, row, form, interp_names, col_prefix: str) -> None:
    """Track running best regret from one history row (obs+fant columns only)."""
    for name in interp_names:
        col = f"{col_prefix}_{name}"
        if name in form.signed_params:
            incumbents[name] = min(incumbents[name], row[col], key=abs)
        else:
            incumbents[name] = min(incumbents[name], row[col])


def run_acquisition(
    args, form, gp_features, train_df, global_best_params, val_best_params, fit_grid, heldout_metric,
):
    """The shared acquisition loop.

    ``fit_grid(df_pred)`` extracts + fits (task-specific); ``heldout_metric(params)`` returns extra
    per-iteration history fields (task-specific). Returns
    ``(history, incumbents, incumbents_val, interp_names, df_pred, acquired_idx, surrogate_pred_rows)``.
    Regrets are logged w.r.t. both the global reference (all data) and the validation reference
    (acquire pool only): ``regret_*`` / ``regret_val_*`` (+ obs-only variants).
    """
    if args.niters is None and args.C_budget is None:
        raise ValueError("Either --niters or --C_budget must be specified.")
    if args.niters is not None and args.C_budget is not None:
        raise ValueError("Only one of --niters or --C_budget can be specified.")

    seed_everything(args.seed)
    sel_kwargs = {"kappa": args.lcb_kappa} if args.selection == "lcb" else {}
    splitter = SELECTION[args.selection](seed=args.seed, **sel_kwargs)
    if args.model_type == "none" and getattr(splitter, "needs_surrogate", False):
        raise ValueError(
            f"--model_type none (model-free) is incompatible with surrogate-based selection "
            f"'{args.selection}' (it needs the GP to pick points). Use a surrogate-free selector, "
            f"e.g. --selection random."
        )
    rng = np.random.RandomState(args.seed)
    y_target = args.y_target

    init_best_hp_idx = None

    if args.init_practice:
        acquired_idx, practice_pool_idx = _init_acquired_idx_practice(
            train_df, args.init_points, args.init_practice_ds, y_target, rng,
        )
        C_lim = float(train_df.loc[acquired_idx, "C"].max())
        init_compute = float(train_df.loc[acquired_idx, "C"].sum())
        k = min(args.init_practice_ds, train_df.loc[practice_pool_idx, "D"].nunique())
        print(
            f"Init (practice): sampled {len(acquired_idx)}/{len(practice_pool_idx)} rows from "
            f"min N={train_df.loc[practice_pool_idx, 'N'].iloc[0]:.3e}, {k} smallest D(s); "
            f"init compute={init_compute:.2e}, C_lim={C_lim:.2e}"
        )
        init_best_hp_idx = acquired_idx.copy()
        n_init = len(acquired_idx)
    else:
        C_lim = train_df.C.min() * args.init_compute_factor
        acquired_idx = _init_acquired_idx_default(
            train_df, args.init_points, rng, C_lim, args.C_window
        )
        init_best_hp_idx = acquired_idx.copy()
        init_compute = float(train_df.loc[acquired_idx, "C"].sum())
        print(f"Init (default): {len(acquired_idx)} random points under C_lim={C_lim:.2e}")
        n_init = len(acquired_idx)

    init_points_idx = init_best_hp_idx.copy()
    n_iters = n_init + (args.niters or (MAX_ITERS - args.init_points))
    total_compute_spent = init_compute

    history = []
    df_pred = None
    surrogate_pred_rows = []
    interp_names = list(form.interpretable(global_best_params).keys())
    global_interp = form.interpretable(global_best_params)
    val_interp = form.interpretable(val_best_params)
    incumbents = {name: np.inf for name in interp_names}
    incumbents_val = {name: np.inf for name in interp_names}
    # incumbent parametric fit (lowest in-data fit loss), used to warm-start the GP mean
    _best_fit_loss, _best_fit_params = np.inf, None

    def _gp_kwargs():
        mk = {}
        if args.gp_mean_warmstart and _best_fit_params is not None:
            mk["init_params"] = _best_fit_params
        return dict(kernel=args.gp_kernel, mean_form=args.gp_mean_form,
                    mean_kwargs=mk, diagnose=args.diagnose_gp, seed=args.seed)
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
                sel_surrogate = MODELS[args.model_type](**_gp_kwargs())
                sel_surrogate.fit(np.log(_obs[gp_features].to_numpy()), _obs[y_target].to_numpy()[:, None])
            except Exception as e:
                # GP fit can raise NotPSDError: the kernel matrix is ill-conditioned (no inverse --
                # a kernel must be positive-definite), which happens with few / near-duplicate points.
                # We fall back to a random pick for this one iteration; warn so that's not silent.
                warnings.warn(
                    f"[iter {iter_idx - n_init}] selection GP fit failed "
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
        pred_y, heldout_mse, _model = build_surrogate(
            np.log(observations[gp_features].to_numpy()),
            observations[y_target].to_numpy()[:, None],
            np.log(train_df.loc[remaining_idx, gp_features].to_numpy()),
            test_y=train_df.loc[remaining_idx, y_target].to_numpy(),
            model_type=args.model_type,
            **_gp_kwargs(),
        )
        if args.model_type != "none" and len(remaining_idx) > 0:
            pred_flat = np.asarray(pred_y, dtype=float).reshape(-1)
            true_y = train_df.loc[remaining_idx, y_target].to_numpy(dtype=float)
            for idx, pred, true in zip(remaining_idx, pred_flat, true_y, strict=True):
                surrogate_pred_rows.append({
                    "iter": iter_idx,
                    "idx": int(idx),
                    "pred_loss": float(pred),
                    "true_loss": float(true),
                })
        # true loss where observed, predicted loss elsewhere, then fit the mixed data.
        # NaN predictions (model-free) drop out, leaving the observed-only fit.
        df_pred = train_df.copy()
        df_pred.loc[remaining_idx, y_target] = pred_y
        df_pred = df_pred.dropna(subset=[y_target])
        best_params, best_loss = fit_grid(df_pred)
        if best_loss < _best_fit_loss:  # incumbent by in-data fit loss, for GP mean warm-start
            _best_fit_loss, _best_fit_params = best_loss, best_params
        print(f"#Iteration {iter_idx} on {observations.shape[0]} obs + {len(remaining_idx)} fant: Fitting loss={best_loss:2.2e}", end="; ")
        print_fit_results(form, best_params)

        # regrets in interpretable space vs global + validation (pool-only) references
        interp = form.interpretable(best_params)
        regrets = _param_regrets(interp, global_interp, form, interp_names)
        regrets_val = _param_regrets(interp, val_interp, form, interp_names)

        if args.model_type == "none":
            # model-free: no infill, so obs-only fit equals obs+fant fit
            obs_only_params, obs_only_best_loss = best_params, best_loss
            obs_only_regrets = regrets
            obs_only_regrets_val = regrets_val
        else:
            obs_only_params, obs_only_best_loss = fit_grid(observations)
            obs_only_interp = form.interpretable(obs_only_params)
            obs_only_regrets = _param_regrets(obs_only_interp, global_interp, form, interp_names)
            obs_only_regrets_val = _param_regrets(obs_only_interp, val_interp, form, interp_names)
            print(f"Fit on {observations.shape[0]} obs only: Fitting loss={obs_only_best_loss:2.2e}", end="; ")
            print_fit_results(form, obs_only_params)

        history.append({
            "iter": iter_idx,
            "n_acquired": len(acquired_idx),
            "acquired_idx": list(acquired_idx),
            "new_idx": int(new_idx),
            "C_lim": float(C_lim),
            "iter_compute": float(iter_compute),
            "total_compute_spent": float(total_compute_spent),
            "heldout_mse": float(heldout_mse),
            "heldout_size": len(remaining_idx),
            **heldout_metric(best_params),
            "params": list(best_params), # obs+fant params at iteration i, not incumbent!
            "fit_loss": float(best_loss),
            "obs_only_params": list(obs_only_params),
            "obs_only_fit_loss": float(obs_only_best_loss),
            **{f"regret_{name}": float(regrets[name]) for name in interp_names},
            **{f"regret_obs_only_{name}": float(obs_only_regrets[name]) for name in interp_names},
            **{f"regret_val_{name}": float(regrets_val[name]) for name in interp_names},
            **{f"regret_obs_only_val_{name}": float(obs_only_regrets_val[name]) for name in interp_names},
        })
        _update_incumbents(incumbents, history[-1], form, interp_names, "regret")
        _update_incumbents(incumbents_val, history[-1], form, interp_names, "regret_val")

        # update C_lim envelope for sampling: ceiling stays one growth step above the max cost seen
        C_lim = observations.C.max()

    print(f"{'='*30}\nIncumbent regrets (global reference):")
    for name in interp_names:
        tag = " (signed)" if name in form.signed_params else ""
        print(f"{name}{tag}: {incumbents[name]:.4f}")
    print(f"{'-'*30}\nIncumbent regrets (validation / pool reference):")
    for name in interp_names:
        tag = " (signed)" if name in form.signed_params else ""
        print(f"{name}{tag}: {incumbents_val[name]:.4f}")
    print(f"{'='*30}")
    return history, incumbents, incumbents_val, interp_names, df_pred, acquired_idx, surrogate_pred_rows


# --- shared plotting helpers (both tasks lay out a grid of per-param regret panels) ---

def _running_incumbent(vals, signed: bool) -> np.ndarray:
    """Running best regret (signed: closest to zero; unsigned: minimum)."""
    if signed:
        running, best = [], vals[0]
        for v in vals:
            if abs(v) < abs(best):
                best = v
            running.append(best)
        return np.asarray(running)
    return np.minimum.accumulate(vals)


def plot_regret_panels(flat_axes, start_idx, history_df, form, interp_names):
    """
    Draw one incumbent-regret panel per interpretable param (validation reference only).

    Colors: red = obs+fant, green = obs-only. Global regrets remain in history for
    post-hoc plotting.
    """
    idx = start_idx
    for name in interp_names:
        ax = flat_axes[idx]
        signed = name in form.signed_params
        series = [
            (f"regret_val_{name}", "val obs+fant", "red"),
            (f"regret_obs_only_val_{name}", "val obs only", "tab:green"),
        ]
        for col, label, color in series:
            if col not in history_df.columns:
                continue
            running = _running_incumbent(history_df[col].to_numpy(), signed)
            ax.plot(history_df["iter"], running, label=label, color=color, lw=1.5)

        if signed:
            ax.axhline(0, color="gray", lw=0.5)
            title = f"incumbent regret {name} (signed)"
        else:
            ax.set_yscale("log")
            title = f"incumbent regret {name}"
        ax.set_xlabel("Samples")
        ax.set_title(title)
        ax.legend(fontsize=7)

        idx += 1
    return idx


def dataset_name_from_input(input_path: str) -> str:
    """Basename of the input CSV without extension (e.g. ``.../oellm_english.csv`` -> ``oellm_english``)."""
    return Path(input_path).stem


def run_experiment_dir(args) -> Path:
    """Top-level directory for one ``--filename`` experiment."""
    return Path(args.output_dir) / args.filename


def run_seed_dir(args) -> Path:
    """Directory for all timestamped runs sharing filename, dataset, form, and seed."""
    return (
        run_experiment_dir(args)
        / dataset_name_from_input(args.input)
        / args.form
        / f"seed_{args.seed}"
    )


def _new_run_timestamp(seed_dir: Path) -> str:
    """UTC timestamp label for a new run under ``seed_dir``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    while (seed_dir / ts).exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return ts


def allocate_run_output_dir(args) -> Path:
    """Pick a run directory: ``{filename}/{dataset}/{form}/seed_{seed}/{timestamp}/``."""
    seed_dir = run_seed_dir(args)
    run_ts = _new_run_timestamp(seed_dir)
    run_dir = seed_dir / run_ts
    print(f"Writing run to {run_dir}/")
    return run_dir


def find_latest_run_dir(args) -> Path | None:
    """Most recent timestamped run for this filename/dataset/form/seed combo."""
    seed_dir = run_seed_dir(args)
    if not seed_dir.is_dir():
        return None
    matches = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _namespace_to_json(args: Namespace) -> dict:
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def save_ref_fits(form, global_best_params, val_best_params, output_dir: Path) -> None:
    """Write global + validation (pool-only) reference parametric fits."""
    payload = {
        "global": {
            "params": list(np.asarray(global_best_params, dtype=float)),
            "interpretable": form.interpretable(global_best_params),
        },
        "validation": {
            "params": list(np.asarray(val_best_params, dtype=float)),
            "interpretable": form.interpretable(val_best_params),
        },
    }
    path = output_dir / "ref_fits.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved reference fits to {path}")


def save_surrogate_predictions(rows, output_dir: Path) -> None:
    """Write unacquired-grid surrogate predictions (long format: iter, idx, pred/true loss)."""
    if not rows:
        return
    path = output_dir / "surrogate_predictions.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"Saved surrogate predictions to {path}")


def save_run_args(args, output_dir: Path) -> None:
    """Write the parsed CLI args (and raw argv) for reproducibility."""
    payload = {
        "task": args.task,
        "argv": sys.argv,
        "args": _namespace_to_json(args),
        "run_dir": str(output_dir),
        "experiment_dir": str(run_experiment_dir(args)),
        "seed_dir": str(output_dir.parent),
        "experiment_name": args.filename,
        "run_timestamp": output_dir.name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "args.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved args to {path}")


def save_panels(fig, flat_axes, panel_idx, path):
    """Blank any unused trailing axes, tidy, and write the figure."""
    for ax in flat_axes[panel_idx:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved figure to {path}")
