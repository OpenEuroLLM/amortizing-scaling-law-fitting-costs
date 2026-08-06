import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from scaling.utils import fit_parametric_form, get_pareto_frontier

from models import GPModel
from selection import RandomSelection, LocalSearch
from parametric_forms import PARAMETRIC_FORMS


SELECTION = {
    "random": RandomSelection,
    "local_search": LocalSearch,
}

MODELS = {
    "gp": GPModel,
}

INPUT_X = ['N', 'D', 'lr', 'gbsz']
OUTPUT_Y = 'loss'

MAX_ITERS = 1000


def next_batch_compute(
    df: pd.DataFrame,
    acquired_idx: pd.Index | None = None,
    C_max: float | None = None,
    C_window: tuple[float, float] | None = None,
    C_envelope_growth: float = 2.0
) -> pd.DataFrame:
    """Subsets the data within a compute bracket.
    """
    pool = df.index.difference(acquired_idx) if acquired_idx is not None else df.index
    if C_max is not None and C_window is not None:
        while True:
            _idxs = (df.loc[pool, "C"] >= C_window[0] * C_max) & (df.loc[pool, "C"] <= C_window[1] * C_max)
            if _idxs.any():
                pool = pool[_idxs]
                break
            elif C_max > df.loc[pool, "C"].max():
                # if the max compute is already above the envelope, just return the pool
                break
            else:
                # push the envelope to not return an empty pool
                C_max *= C_envelope_growth
    return df.loc[pool]


def data_preprocess(
    df: pd.DataFrame,
    held_out_type: str = 'N',
    held_out_C_lower: float = 0.5,
    C_extrapolation_len: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["C"] = 6 * df["N"] * df["D"]

    # Keep the max compute as the test point
    if held_out_type == 'N':
        test_df_known = df.loc[df.N == df.N.max()]
        _upper_lim, _col = df.N.max(), "N"
    elif held_out_type == 'C':
        _pseudo_C_max = df.C.max() * held_out_C_lower
        test_df_known = df.loc[df.C >= _pseudo_C_max]
        _upper_lim, _col = _pseudo_C_max, "C"
    else:
        raise ValueError(f"Unsupported held_out_type: {held_out_type}")
    
    # Create an extrapolation limit as unknown test set
    test_df_unknown = pd.DataFrame.from_dict({
        "C": [test_df_known.C.max() * 10 ** i for i in range(1, C_extrapolation_len + 1)],
        "N": [None] * C_extrapolation_len,
        "D": [None] * C_extrapolation_len,
        "lr": [None] * C_extrapolation_len,
        "gbsz": [None] * C_extrapolation_len,
        "loss": [None] * C_extrapolation_len,
    })

    train_df = df.loc[df[_col] < _upper_lim]
    
    return train_df, test_df_known, test_df_unknown


def iterative_data_splits(
    df: pd.DataFrame,
    splitter: RandomSelection | LocalSearch,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, float, pd.Index]:
    X, y, total_compute, remaining_idx = splitter.split(df, n_points)
    return X, y, total_compute, remaining_idx


def build_surrogate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray | None = None,
    model_type: str = "gp",
    **kwargs
) -> tuple[np.ndarray, float]:
    if model_type in MODELS:
        model = MODELS[model_type](**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(train_x, train_y, **kwargs)
    pred_y = model.predict(test_x)
    mse = None
    if test_y is not None:
        mse = np.mean((test_y - pred_y.flatten()) ** 2)

    return pred_y, mse


def build_fit(
    df: pd.DataFrame,
    form_name: str = "L0",
    y_name: str | None = None,
    fit_quality: str = 'debug',
) -> tuple[np.ndarray, float]:
    form = PARAMETRIC_FORMS[form_name]
    y_name = y_name if y_name is not None else form.target
    fit_df = get_pareto_frontier(df, x_name=form.input_cols[0], y_name=y_name) if form.use_pareto else df
    X_data = fit_df[form.input_cols].to_numpy()
    if X_data.shape[1] == 1:
        X_data = X_data.ravel()
    best_params, best_loss, _ = fit_parametric_form(
        form.func,
        X_data,
        np.log(fit_df[y_name].to_numpy()),
        form.initial_grid(fit_quality),
        form.bounds,
    )
    return best_params, best_loss


def _print_fit_results(form, params: np.ndarray):
    print(form.format_fit(params))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--niters', type=int, default=None)
    parser.add_argument('--total_C', type=float, default=None)
    parser.add_argument('--init_points', type=int, default=25)
    parser.add_argument(
        '--init_compute_factor', type=float, default=10.0,
        help="Factor to multiply the minimum compute cost to set the initial envelope for sampling."
    )
    parser.add_argument(
        '--C_window', type=float, nargs=2, default=[1e-3, 1e3], metavar=('LO', 'HI'),
        help="Factors on the max. seen compute to determine permissible range for sampling."
    )
    parser.add_argument('--model_type', type=str, default='gp', choices=['gp'])
    parser.add_argument('--selection', type=str, default='random', choices=list(SELECTION))
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        '--form', type=str, default='L0', choices=list(PARAMETRIC_FORMS),
        help="Parametric functional form to fit. 'L0': L(C) = L0 + A*C^alpha. "
        "'chinchilla3': L(N,D) = E + A/N^alpha + B/D^beta (Chinchilla Approach 3). "
        "'chinchilla_interaction': chinchilla3 plus an N*D interaction term with its own "
        "single exponent, L(N,D) = E + A/N^alpha + B/D^beta + G/(N*D)^gamma. "
        "'width_depth': L(width,depth,D) = E + A/width^alpha + B/depth^beta + F/D^gamma. "
        "'steplaw_lr'/'steplaw_bsz': Step Law (arXiv:2503.04715) optimal LR/batch-size "
        "power laws -- build_fit() fits these against 'lr'/'gbsz' by default (form.target); "
        "they have no L-vs-C marginalization, so don't drive them through this script's "
        "C-extrapolation path."
    )
    parser.add_argument(
        '--fit_quality', type=str, default='debug', choices=['debug', 'dense', 'super_dense']
    )
    parser.add_argument('--filename', default="history")
    parser.add_argument('--plot_figure', action='store_true')

    parser.add_argument(
        '--held_out_type', type=str, default='N', choices=['N', 'C'],
        help="Whether to hold out the largest N or the largest C for testing. Default is 'N'."
    )
    parser.add_argument(
        '--held_out_C_lower', type=float, default=0.5,
        help="held_out_C_lower*C_min defines the threshold for the held-out set when "\
        "--held_out_type is 'C'. Default is 0.5."
    )
    parser.add_argument(
        '--gp_mean_as_l0', action='store_true', 
        help="If set, the GP mean function is initialized with the previous iteration's best-fit L0. " \
        "If not set, the GP mean is a constant function (default)."
    )
    parser.add_argument(
        '--diagnose_gp', action='store_true',
        help="If set, prints GP diagnostics (MLL, train MSE, noise, lengthscales, mean parameters) after fitting."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    # resolve runtime budget
    if args.niters is None and args.total_C is None:
        raise ValueError("Either --niters or --total_C must be specified.")
    if args.niters is not None and args.total_C is not None:
        raise ValueError("Only one of --niters or --total_C can be specified.")
    
    total_compute_spent = 0
    total_iters = 0
    n_iters = args.init_points + (args.niters or (MAX_ITERS - args.init_points))

    # read data
    df = pd.read_csv(args.input)

    # preprocess data
    train_df, test_df_known, test_df_unknown = data_preprocess(
        df, held_out_type=args.held_out_type, held_out_C_lower=args.held_out_C_lower
    )

    form = PARAMETRIC_FORMS[args.form]

    # ground truth prep
    _params_so_far = (0.7, 2.0, -0.05)
    _global_df = pd.concat((train_df, test_df_known))
    print("\nBuilding global surrogate on all data as a baseline...")
    global_pred, global_mse = build_surrogate(
        np.log(_global_df[INPUT_X].to_numpy()),
        _global_df[OUTPUT_Y].to_numpy()[:, None],
        np.log(test_df_known[INPUT_X].to_numpy()),
        model_type=args.model_type,
        use_mean_model=True if args.gp_mean_as_l0 else False,
        diagnose=args.diagnose_gp,
    )
    print("Fitting parametric form on the global surrogate...")
    global_best_params, global_best_loss = build_fit(
        _global_df, form_name=args.form, y_name="loss", fit_quality=args.fit_quality
    )
    _print_fit_results(form, global_best_params)

    # test targets for per-iteration fit evaluation
    _test_pareto = get_pareto_frontier(test_df_known, x_name="C", y_name="loss")
    C_known = _test_pareto["C"].to_numpy()
    L_known = _test_pareto["loss"].to_numpy()
    C_unknown = test_df_unknown["C"].to_numpy().astype(float)

    # main loop here
    splitter = SELECTION[args.selection](seed=args.seed)
    rng = np.random.RandomState(args.seed)

    # initialization: seed with the cheapest point, then fill up to init_points under the envelope
    C_lim = train_df.C.min() * args.init_compute_factor
    total_compute_spent = 0
    pool = next_batch_compute(train_df, acquired_idx=None, C_max=C_lim, C_window=args.C_window)
    acquired_idx = pd.Index(rng.choice(pool.index, args.init_points , replace=False))
    total_compute_spent = train_df.loc[acquired_idx, "C"].sum()

    # main loop
    history = []
    interp_names = list(form.interpretable(global_best_params).keys())
    incumbents = {name: np.inf for name in interp_names}
    for iter_idx in range(args.init_points, n_iters + 1):
        # budget checks
        if args.total_C is not None and total_compute_spent >= args.total_C:
            print(f"\n\n{'=' * 50}\nReached total compute budget of {args.total_C}. Stopping iterations.\n{'=' * 50}\n")
            break
        if len(acquired_idx) == len(train_df):
            print(f"\n\n{'=' * 50}\nAll {len(train_df)} points acquired. Stopping iterations.\n{'=' * 50}\n")
            break

        # sample a new point under the cost envelope; its C is this iteration's cost
        pool = next_batch_compute(train_df, acquired_idx, C_max=C_lim, C_window=args.C_window)
        new_idx = splitter.next_point(train_df, pool.index)
        acquired_idx = acquired_idx.append(pd.Index([new_idx]))
        iter_compute = train_df.loc[new_idx, "C"]

        # add compute spent
        total_compute_spent += iter_compute

        observations = train_df.loc[acquired_idx]
        remaining_idx = train_df.index.difference(acquired_idx)

        # surrogate: fit on observations (log-features), predict the unacquired rest
        pred_y, heldout_mse = build_surrogate(
            np.log(observations[INPUT_X].to_numpy()),
            observations[OUTPUT_Y].to_numpy()[:, None],
            np.log(train_df.loc[remaining_idx, INPUT_X].to_numpy()),
            test_y=train_df.loc[remaining_idx, OUTPUT_Y].to_numpy(),
            model_type=args.model_type,
            use_mean_model=True if args.gp_mean_as_l0 else False,
            diagnose=args.diagnose_gp
        )

        # true loss where observed, predicted loss elsewhere
        df_pred = train_df.copy()
        df_pred.loc[remaining_idx, OUTPUT_Y] = pred_y

        # parametric fit on the mixed data
        best_params, best_loss = build_fit(
            df_pred[INPUT_X + ["C", OUTPUT_Y]], form_name=args.form, fit_quality=args.fit_quality
        )
        print(f"#Iteration {iter_idx}: Fitting loss={best_loss:2.2e}", end="; ")
        _print_fit_results(form, best_params)
        _params_so_far = best_params  # for next iteration's GP mean initialization if requested

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
            "known_test_mse": float(np.mean(
                (form.to_L_vs_C(C_known, best_params) - L_known) ** 2
            )),
            "L_known": list(form.to_L_vs_C(C_known, best_params)),
            "C_unknown": list(C_unknown),
            "L_unknown": list(form.to_L_vs_C(C_unknown, best_params)),
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

        # # update C_lim envelope for sampling: ceiling stays one growth step above the max cost seen
        C_lim = observations.C.max()
    print(f"{'-'*30}")
    print("Global surrogate fit results:")
    _print_fit_results(form, global_best_params)
    print(f"Known Test MSE: {history[-1]['known_test_mse']:.2e}")
    print(f"Unknown Test Predictions:")
    for i in range(len(C_unknown)):
        print(f"    C={C_unknown[i]:.2e} -> L={history[-1]['L_unknown'][i]:.3f}")
    print(f"{'-'*30}")
    print("Run done!")

    print(f"{'='*30}\nIncumbent regrets:")
    for name in interp_names:
        tag = " (signed)" if name in form.signed_params else ""
        print(f"{name}{tag}: {incumbents[name]:.4f}")
    print(f"{'='*30}")

    # saving to disk
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_parquet(output_dir / f"{args.filename}.parquet", index=False)
    print(f"Saved history to {output_dir / f'{args.filename}.parquet'}")


    if args.plot_figure:

        # panels: one regret plot per interpretable param of `form`, plus known-test mse,
        # compute/iter, and the fit-comparison plot. Laid out dynamically since the param
        # count (and hence panel count) varies by form.
        n_extra_panels = 3  # known-test mse, compute/iter, fit-comparison
        total_panels = len(interp_names) + n_extra_panels
        ncols = 3
        nrows = int(np.ceil(total_panels / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
        flat_axes = np.atleast_2d(axes).flatten()

        panel_idx = 0
        for name in interp_names:
            ax = flat_axes[panel_idx]
            col = f"regret_{name}"
            if name in form.signed_params:
                # running incumbent closest to zero, sign preserved (matches printed incumbents)
                vals = history_df[col].to_numpy()
                signed_running = []
                best = vals[0]
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
            panel_idx += 1

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

        # fit-comparison panel: real vs fantasized parametric fits, plotted as L vs C
        ax = flat_axes[panel_idx]
        panel_idx += 1

        # "real": fit only on actually observed points (init + acquired)
        obs_df = train_df.loc[acquired_idx]
        real_params, _ = build_fit(obs_df[INPUT_X + ["C", OUTPUT_Y]], form_name=args.form, fit_quality=args.fit_quality)
        # "fantasized": last fit from history (true observed + GP-predicted rest)
        fant_params = history_df.iloc[-1]["params"]

        # scatter: seen pareto + known-test pareto (unknown has no ground truth)
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
        ax.set_xlabel("C")
        ax.set_ylabel("loss")
        ax.legend(fontsize=15)
        ax.set_title("real vs fantasized fit (dashed = beyond seen C)")

        # hide any unused trailing axes
        for ax in flat_axes[panel_idx:]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / f"{args.filename}.png", dpi=150)
        print(f"Saved figure to {output_dir / f'{args.filename}.png'}")

        print(f"{'='*30}\nDiagnostics for Pareto membership:")
        # pareto from the observations
        obs_pareto  = get_pareto_frontier(obs_df,  x_name="C", y_name="loss")
        acq = set(obs_pareto.index)  # indices of observed points
        # pareto from the observations + GP-predicted rest (fantasized)
        fant_pareto = get_pareto_frontier(df_pred, x_name="C", y_name="loss")
        fant_idx = set(fant_pareto.index)  # indices of fantasized pareto points
        labels = [
            "observed-only frontier", 
            "fantasized frontier", 
            "    from observed points", 
            "    from GP predictions"
        ]
        width = max(len(label) for label in labels)
        print(f"{labels[0]:<{width}} : {len(obs_pareto)} points")
        print(f"{labels[1]:<{width}} : {len(fant_pareto)} points")
        print(f"{labels[2]:<{width}} : {len(fant_idx & acq)} points")
        print(f"{labels[3]:<{width}} : {len(fant_idx - acq)} points")
# end of file