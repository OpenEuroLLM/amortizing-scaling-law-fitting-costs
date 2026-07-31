import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from scaling.utils import fit_parametric_form, functional_form_chin3, get_pareto_frontier

from models import GPModel
from selection import RandomSelection, LocalSearch
from iterative_fits_L0_form import (
    build_surrogate, data_preprocess, iterative_data_splits, next_batch_compute
)


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


def param_fit_setup_args(
    fit_quality: str = 'debug',
) -> tuple[list[list[float]], list[tuple[float, float]]]:
    """Initial grid and bounds for the Chinchilla Approach-3 form
    L(N, D) = A/N^alpha + B/D^beta + E, parameterized as [a, alpha, b, beta, e]
    with a = log(A), b = log(B), e = log(E). Note alpha, beta are POSITIVE here
    (the form divides by N^alpha), unlike the L0-form's negative exponent.
    """
    if fit_quality == 'debug':  # 2 x 3 x 2 x 3 x 2 = 72
        initial_grid = [
            [a, alpha, b, beta, e]
            for a in [2.5, 7.5]                  # a = log(A)
            for alpha in [0.2, 0.5, 1.0]
            for b in [2.5, 7.5]                  # b = log(B)
            for beta in [0.2, 0.5, 1.0]
            for e in [0.0, 0.7]                  # e = log(E), E ~ 1..2
        ]
    elif fit_quality == 'dense':  # 4 x 4 x 4 x 4 x 3 = 768
        initial_grid = [
            [a, alpha, b, beta, e]
            for a in [0., 5., 10., 15.]
            for alpha in [0.1, 0.3, 0.7, 1.2]
            for b in [0., 5., 10., 15.]
            for beta in [0.1, 0.3, 0.7, 1.2]
            for e in [-0.5, 0.3, 0.7]
        ]
    elif fit_quality == 'super_dense':  # 6 x 5 x 6 x 5 x 5 = 4500 (Chinchilla paper grid)
        initial_grid = [
            [a, alpha, b, beta, e]
            for a in [0., 5., 10., 15., 20., 25.]
            for alpha in [0., 0.5, 1., 1.5, 2.]
            for b in [0., 5., 10., 15., 20., 25.]
            for beta in [0., 0.5, 1., 1.5, 2.]
            for e in [-1., -0.5, 0., 0.5, 1.]
        ]
    else:
        raise ValueError(f"Unsupported fit quality: {fit_quality}")
    bounds = [
        (0.0, 30.0),                  # a = log(A)
        (0.0, 2.5),                   # alpha (positive)
        (0.0, 30.0),                  # b = log(B)
        (0.0, 2.5),                   # beta (positive)
        (np.log(1e-3), np.log(4.0)),  # e = log(E), irreducible loss
    ]
    return initial_grid, bounds


def build_fit(
    df: pd.DataFrame,
    x_names: tuple[str, str] = ("N", "D"),
    y_name: str = "loss",
    fit_quality: str = 'debug',
) -> tuple[np.ndarray, float]:
    """Approach-3 fit: no Pareto frontier. For each (N, D) cell, take the best
    observed/predicted loss (marginalizing over lr, gbsz), then fit L(N, D).
    """
    df_best = df.loc[df.groupby(list(x_names))[y_name].idxmin()]
    best_params, best_loss, _ = fit_parametric_form(
        functional_form_chin3,
        df_best[list(x_names)].to_numpy(),          # 2D: columns (N, D), raw scale
        np.log(df_best[y_name].to_numpy()),         # func returns log(L)
        *param_fit_setup_args(fit_quality)
    )
    return best_params, best_loss


def _print_fit_results(params: np.ndarray):
    assert len(params) == 5, "Expected 5 parameters: a, alpha, b, beta, e"
    a, alpha, b, beta, e = params
    print(f"L(N,D) = {np.exp(a):.3e}/N^{alpha:.3f} + {np.exp(b):.3e}/D^{beta:.3f} + {np.exp(e):.3f}")


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
        '--gp_mean_warmstart', action='store_true',
        help="Initialize the GP mean with the best parametric fit seen so far "
        "(lowest in-data fit loss). Only relevant when --gp_mean_form is set."
    )
    parser.add_argument(
        '--gp_mean_form', type=str, default=None, choices=['l0', 'chin3'],
        help="Parametric GP mean to use ('chin3' matches this script's fit form). "
        "If omitted, the GP mean is a constant function (default)."
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
    # incumbent parametric fit (lowest in-data fit loss), used to warm-start the GP mean
    _best_fit_loss, _best_fit_params = np.inf, None
    _global_df = pd.concat((train_df, test_df_known))
    print("\nBuilding global surrogate on all data as a baseline...")
    global_pred, global_mse = build_surrogate(
        np.log(_global_df[INPUT_X].to_numpy()),
        _global_df[OUTPUT_Y].to_numpy()[:, None],
        np.log(test_df_known[INPUT_X].to_numpy()),
        model_type=args.model_type,
        mean_form=args.gp_mean_form,
        diagnose=args.diagnose_gp,
    )
    print("Fitting parametric form on the global surrogate...")
    global_best_params, global_best_loss = build_fit(
        _global_df, y_name="loss", fit_quality=args.fit_quality
    )
    _print_fit_results(global_best_params)

    # test targets for per-iteration fit evaluation (Approach 3: best loss per (N, D) cell)
    _test_best = test_df_known.loc[test_df_known.groupby(["N", "D"])["loss"].idxmin()]
    ND_known = _test_best[["N", "D"]].to_numpy()
    C_known = _test_best["C"].to_numpy()
    L_known = _test_best["loss"].to_numpy()
    # NOTE: chin3 predicts from (N, D), not C.

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
    incumbents = {k: np.inf for k in ("E", "A", "alpha", "B", "beta")}
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
            mean_form=args.gp_mean_form,
            mean_kwargs=(
                {"init_params": _best_fit_params}
                if args.gp_mean_warmstart and _best_fit_params is not None else {}
            ),
            diagnose=args.diagnose_gp
        )

        # true loss where observed, predicted loss elsewhere
        df_pred = train_df.copy()
        df_pred.loc[remaining_idx, OUTPUT_Y] = pred_y

        # parametric fit on the mixed data
        best_params, best_loss = build_fit(df_pred[INPUT_X + [OUTPUT_Y]], fit_quality=args.fit_quality)
        if best_loss < _best_fit_loss:  # incumbent by in-data fit loss
            _best_fit_loss, _best_fit_params = best_loss, best_params
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
                (functional_form_chin3(ND_known, best_params, return_log_loss=False) - L_known) ** 2
            )),
            "L_known": list(functional_form_chin3(ND_known, best_params, return_log_loss=False)),
            "params": list(best_params),
            "fit_loss": float(best_loss),
            # regrets in interpretable space: A = exp(a), B = exp(b), E = exp(e);
            # alpha, beta are the exponents directly. E signed (loss floor over/under-estimate).
            "regret_A": float(np.abs(np.exp(best_params[0]) - np.exp(global_best_params[0]))),
            "regret_alpha": float(np.abs(best_params[1] - global_best_params[1])),
            "regret_B": float(np.abs(np.exp(best_params[2]) - np.exp(global_best_params[2]))),
            "regret_beta": float(np.abs(best_params[3] - global_best_params[3])),
            "regret_E": float(np.exp(best_params[4]) - np.exp(global_best_params[4])),  # signed
        })
        # updating incumbent tracker
        # E regret is signed: incumbent is the value closest to zero, sign preserved
        incumbents["E"] = min(incumbents["E"], history[-1]["regret_E"], key=abs)
        for _k in ("A", "alpha", "B", "beta"):
            incumbents[_k] = min(incumbents[_k], history[-1][f"regret_{_k}"])

        # # update C_lim envelope for sampling: ceiling stays one growth step above the max cost seen
        C_lim = observations.C.max()
    print(f"{'-'*30}")
    print("Global surrogate fit results:")
    _print_fit_results(global_best_params)
    print(f"Known Test MSE: {history[-1]['known_test_mse']:.2e}")
    print(f"{'-'*30}")
    print("Run done!")

    print(f"{'='*30}\nIncumbent regrets:")
    for _k, _v in incumbents.items():
        print(f"{_k}: {_v:.4f}" + (" (signed)" if _k == "E" else ""))
    print('='*30)

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
            (axes[0, 0], "regret_E", "regret E (signed)", None),
            (axes[0, 1], "regret_alpha", "regret alpha", "log"),
            (axes[0, 2], "regret_beta", "regret beta", "log"),
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
        real_params, _ = build_fit(obs_df[INPUT_X + [OUTPUT_Y]], fit_quality=args.fit_quality)
        # "fantasized": last fit from history (true observed + GP-predicted rest)
        fant_params = history_df.iloc[-1]["params"]

        # scatter: best-per-(N,D) of observations + known test
        _obs_best = obs_df.loc[obs_df.groupby(["N", "D"])[OUTPUT_Y].idxmin()]
        ax.scatter(_obs_best["C"], _obs_best["loss"], marker="x", color="black", zorder=3, label="observed")
        ax.scatter(C_known, L_known, marker="x", color="tab:blue", zorder=3, label="held out")

        # predictions at the known-test (N, D) cells from the two fits
        for params, color, marker, label in [
            (real_params, "tab:green", "o", "fit on observed only"),
            (fant_params, "red", "^", "fit on fantasized grid"),
        ]:
            L_hat = functional_form_chin3(ND_known, np.asarray(params), return_log_loss=False)
            ax.scatter(C_known, L_hat, marker=marker, facecolors="none",
                       edgecolors=color, zorder=2, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("C")
        ax.set_ylabel("loss")
        ax.legend(fontsize=12)
        ax.set_title("held-out cells: truth (blue x) vs fit predictions")

        fig.tight_layout()
        fig.savefig(output_dir / f"{args.filename}.png", dpi=150)
        print(f"Saved figure to {output_dir / f'{args.filename}.png'}")

        labels = [
            "Global surrogate", 
            "Observed grid", 
            "  #Best HPs recovered",
            "Fantasized grid",
            "  #Best HPs recovered",
        ]
        width = max(len(label) for label in labels)
        print(f"{'='*30}\nDiagnostics for modeling membership:")
        _global_opt = _global_df.loc[_global_df.groupby(["N", "D"])[OUTPUT_Y].idxmin()]
        print(f"{labels[0]:<{width}} : {len(_global_opt)} points")
        _obs_opt = obs_df.loc[obs_df.groupby(["N", "D"])[OUTPUT_Y].idxmin()]
        print(f"{labels[1]:<{width}} : {len(_obs_opt)} points")
        print(f"{labels[2]:<{width}} : {len(set(_global_opt.index) & set(_obs_opt.index))} points")
        _pred_opt = df_pred.loc[df_pred.groupby(["N", "D"])[OUTPUT_Y].idxmin()]
        print(f"{labels[3]:<{width}} : {len(_pred_opt)} points")
        print(f"{labels[4]:<{width}} : {len(set(_global_opt.index) & set(_pred_opt.index))} points")
# end of file