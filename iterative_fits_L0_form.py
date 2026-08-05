import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from scaling.utils import fit_parametric_form, functional_form_L0, get_pareto_frontier

from models import GPModel
from selection import RandomSelection, LocalSearch, FarthestPointSelection


SELECTION = {
    "random": RandomSelection,
    "local_search": LocalSearch,
    "farthest_point": FarthestPointSelection,
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
    x_name: str = "C",
    y_name: str = "loss",
    fit_quality: str = 'debug',
) -> tuple[np.ndarray, float]:
    pareto_df = get_pareto_frontier(df, x_name=x_name, y_name=y_name)
    best_params, best_loss, _ = fit_parametric_form(
        functional_form_L0,
        pareto_df[x_name].to_numpy(),
        np.log(pareto_df[y_name].to_numpy()),
        *param_fit_setup_args(fit_quality)
    )
    return best_params, best_loss


def param_fit_setup_args(
    fit_quality: str = 'debug',
) -> tuple[list[list[float]], list[tuple[float, float]]]:
    """Initial grid and bounds for parametric fit of L(C) = L0 + A * C^alpha.
    """
    if fit_quality == 'debug': # 3 x 3 x 3 = 27
        initial_grid = [ 
            [l0, a, b]
            for l0 in np.log([0.5, 1.0, 2.0])     # L0 guesses
            for a in [0., 5., 10.,]              # a = log(coefficient)
            for b in [-0.05, -0.2, -0.5]          # exponent (negative!)
        ]
    elif fit_quality == 'dense':  # 6 x 6 x 6 = 216
        initial_grid = [ 
            [l0, a, b]
            for l0 in np.linspace(np.log(0.1), np.log(3.0), 6)
            for a in [0., 5., 10., 20., 30., 40.]
            for b in [-0.02, -0.05, -0.1, -0.2, -0.5, -1.0]
        ]
    elif fit_quality == 'super_dense':   # 10 x 13 x 14 = 1820
        initial_grid = [
            [l0, a, b]
            for l0 in np.linspace(np.log(0.05), np.log(3.5), 8)
            for a in [0., 2.5, 5., 7.5, 10., 15., 20., 25., 35., 45.]
            for b in [-0.01, -0.02, -0.05, -0.08, -0.12, -0.2, -0.35, -0.5, -0.75, -1.0]
        ]
    else:
        raise ValueError(f"Unsupported fit quality: {fit_quality}")
    bounds = [
        (np.log(1e-3), np.log(4.0)),  # L0 bounds
        (None, None),                 # A bounds
        (-1.0, 0.0)                   # alpha bounds
    ]
    return initial_grid, bounds


def _print_fit_results(params: np.ndarray):
    assert len(params) == 3, "Expected 3 parameters: L0, A, alpha"
    L0, A, alpha = params
    print(f"L(C) = {np.exp(L0):.3f} + {np.exp(A):.3f} * C^({alpha:.4f})")


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
    parser.add_argument(
        '--model_free', action='store_true',
        help="Model-free mode: fit L(C) directly on the observed compute-optimal points, skipping "
        "the GP surrogate grid-infill. Turns e.g. --selection random into a fully model-free baseline."
    )
    parser.add_argument('--selection', type=str, default='random', choices=list(SELECTION))
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
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
        _global_df, x_name="C", y_name="loss", fit_quality=args.fit_quality
    )
    _print_fit_results(global_best_params)

    # test targets for per-iteration fit evaluation
    _test_pareto = get_pareto_frontier(test_df_known, x_name="C", y_name="loss")
    C_known = _test_pareto["C"].to_numpy()
    L_known = _test_pareto["loss"].to_numpy()
    C_unknown = test_df_unknown["C"].to_numpy().astype(float)

    # main loop here
    splitter = SELECTION[args.selection](seed=args.seed)
    # model-free skips the surrogate entirely; a surrogate-based selector can't run without it.
    if args.model_free and getattr(splitter, "needs_surrogate", False):
        raise ValueError(
            f"--model_free is incompatible with surrogate-based selection '{args.selection}' "
            f"(it needs the GP to pick points). Use a surrogate-free selector, e.g. --selection random."
        )
    rng = np.random.RandomState(args.seed)

    # initialization: seed with the cheapest point, then fill up to init_points under the envelope
    C_lim = train_df.C.min() * args.init_compute_factor
    total_compute_spent = 0
    pool = next_batch_compute(train_df, acquired_idx=None, C_max=C_lim, C_window=args.C_window)
    acquired_idx = pd.Index(rng.choice(pool.index, args.init_points , replace=False))
    total_compute_spent = train_df.loc[acquired_idx, "C"].sum()

    # main loop
    history = []
    incumbents = {
        "l0": np.inf,
        "A": np.inf,
        "alpha": np.inf,
    }
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
        new_idx = splitter.next_point(train_df, pool.index, acquired_idx=acquired_idx)
        acquired_idx = acquired_idx.append(pd.Index([new_idx]))
        iter_compute = train_df.loc[new_idx, "C"]

        # add compute spent
        total_compute_spent += iter_compute

        observations = train_df.loc[acquired_idx]
        remaining_idx = train_df.index.difference(acquired_idx)

        if args.model_free:
            # model-free: fit L(C) directly on the observed points, no surrogate infill
            heldout_mse = float("nan")
            best_params, best_loss = build_fit(
                observations[INPUT_X + ["C", OUTPUT_Y]], fit_quality=args.fit_quality)
        else:
            # surrogate: fit a GP on observations (log-features), predict the unacquired rest
            pred_y, heldout_mse = build_surrogate(
                np.log(observations[INPUT_X].to_numpy()),
                observations[OUTPUT_Y].to_numpy()[:, None],
                np.log(train_df.loc[remaining_idx, INPUT_X].to_numpy()),
                test_y=train_df.loc[remaining_idx, OUTPUT_Y].to_numpy(),
                model_type=args.model_type,
                use_mean_model=True if args.gp_mean_as_l0 else False,
                diagnose=args.diagnose_gp
            )
            # true loss where observed, predicted loss elsewhere, then fit the mixed data
            df_pred = train_df.copy()
            df_pred.loc[remaining_idx, OUTPUT_Y] = pred_y
            best_params, best_loss = build_fit(df_pred[INPUT_X + ["C", OUTPUT_Y]], fit_quality=args.fit_quality)
        print(f"#Iteration {iter_idx}: Fitting loss={best_loss:2.2e}", end="; ")
        _print_fit_results(best_params)
        _params_so_far = best_params  # for next iteration's GP mean initialization if requested

        history.append({
            "iter": iter_idx,
            "n_acquired": len(acquired_idx),
            "acquired_idx": int(new_idx),
            "iter_compute": float(iter_compute),
            "total_compute_spent": float(total_compute_spent),
            "heldout_mse": float(heldout_mse),
            "heldout_size": len(remaining_idx),
            "known_test_mse": float(np.mean(
                (functional_form_L0(C_known, best_params, return_log_loss=False) - L_known) ** 2
            )),
            "L_known": list(functional_form_L0(C_known, best_params, return_log_loss=False)),
            "C_unknown": list(C_unknown),
            "L_unknown": list(functional_form_L0(C_unknown, best_params, return_log_loss=False)),
            "params": list(best_params),
            "fit_loss": float(best_loss),
            # regrets in interpretable space: L0 = exp(l0), A = exp(a); alpha = b directly
            "regret_L0": float(np.exp(best_params[0]) - np.exp(global_best_params[0])),  # signed
            "regret_A": float(np.abs(np.exp(best_params[1]) - np.exp(global_best_params[1]))),
            "regret_alpha": float(np.abs(best_params[2] - global_best_params[2])),
        })
        # updating incumbent tracker
        # l0 regret is signed: incumbent is the value closest to zero, sign preserved
        incumbents["l0"] = min(incumbents["l0"], history[-1]["regret_L0"], key=abs)
        incumbents["A"] = min(incumbents["A"], history[-1]["regret_A"])
        incumbents["alpha"] = min(incumbents["alpha"], history[-1]["regret_alpha"])

        # # update C_lim envelope for sampling: ceiling stays one growth step above the max cost seen
        C_lim = observations.C.max()
    print(f"{'-'*30}")
    print("Global surrogate fit results:")
    _print_fit_results(global_best_params)
    print(f"Known Test MSE: {history[-1]['known_test_mse']:.2e}")
    print(f"Unknown Test Predictions:")
    for i in range(len(C_unknown)):
        print(f"    C={C_unknown[i]:.2e} -> L={history[-1]['L_unknown'][i]:.3f}")
    print(f"{'-'*30}")
    print("Run done!")

    print(f"{'='*30}\nIncumbent regrets:")
    print(f"L0: {incumbents['l0']:.3f} (signed)\nA: {incumbents['A']:.3f}\nalpha: {incumbents['alpha']:.4f}\n{'='*30}")

    # saving to disk
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_parquet(output_dir / f"{args.filename}.parquet", index=False)
    print(f"Saved history to {output_dir / f'{args.filename}.parquet'}")


    if args.plot_figure:

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        # row 1: regrets; row 2 (first two): known-test mse, cumulative compute
        panels = [
            (axes[0, 0], "regret_L0", "regret L0 (signed)", None),
            (axes[0, 1], "regret_A", "regret A", "log"),
            (axes[0, 2], "regret_alpha", "regret alpha", "log"),
            (axes[1, 0], "known_test_mse", "known-test MSE", "log"),
            (axes[1, 1], "iter_compute", "compute / iter", "log"),
        ]
        for ax, col, title, yscale in panels:
            if "regret" in title:
                ax.plot(history_df["iter"], np.minimum.accumulate(history_df[col]))
            else:
                ax.plot(history_df["iter"], history_df[col])
            if yscale:
                ax.set_yscale(yscale)
            ax.set_xlabel("iteration")
            ax.set_title(title) if "regret" not in title else ax.set_title(f"incumbent {title}")
        axes[0, 0].axhline(0, color="gray", lw=0.5)

        # row 2, third column: real vs fantasized parametric fits
        ax = axes[1, 2]

        # "real": fit only on actually observed points (init + acquired)
        obs_df = train_df.loc[acquired_idx]
        real_params, _ = build_fit(obs_df[INPUT_X + ["C", OUTPUT_Y]], fit_quality=args.fit_quality)
        # "fantasized": last fit from history (true observed + GP-predicted rest)
        fant_params = history_df.iloc[-1]["params"]

        # scatter: seen pareto + known-test pareto (unknown has no ground truth)
        _obs_pareto = get_pareto_frontier(obs_df, x_name="C", y_name="loss")
        ax.scatter(_obs_pareto["C"], _obs_pareto["loss"], marker="x", color="black", zorder=3, label="observed")
        ax.scatter(C_known, L_known, marker="x", color="tab:blue", zorder=3, label="held out")

        C_grid = np.geomspace(obs_df["C"].min(), C_unknown.max(), 200)
        seen = C_grid <= obs_df["C"].max()
        for params, color, label in [(real_params, "tab:green", "subsampled grid"), (fant_params, "red", "fantasized dense grid")]:
            L_hat = functional_form_L0(C_grid, np.asarray(params), return_log_loss=False)
            ax.plot(C_grid[seen], L_hat[seen], color=color, ls="-", label=label)
            ax.plot(C_grid[~seen], L_hat[~seen], color=color, ls="--")
        L_global_hat = functional_form_L0(C_grid, np.asarray(global_best_params), return_log_loss=False)
        ax.plot(C_grid[seen], L_global_hat[seen], color="tab:purple", ls="-.", label="all data fit")
        ax.plot(C_grid[~seen], L_global_hat[~seen], color="tab:purple", ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("C")
        ax.set_ylabel("loss")
        ax.legend(fontsize=15)
        ax.set_title("real vs fantasized fit (dashed = beyond seen C)")

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