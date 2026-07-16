import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from scaling.utils import fit_parametric_form, functional_form_L0, get_pareto_frontier

from models import GPModel
from selection import RandomSelection, LocalSearch


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
    acquired_idx: pd.Index | int,
    C_lim: float | None = None,
    C_envelope_growth: float = 2.0
) -> pd.DataFrame:
    pool = df.index.difference(acquired_idx)
    if C_lim is not None:
        while True:
            _idxs = (df.loc[pool, "C"] <= C_lim)
            if _idxs.any() or C_lim >= df.C.max():
                pool = pool[_idxs]
                break
            else:
                C_lim *= C_envelope_growth
    return df.loc[pool]


def data_preprocess(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["C"] = 6 * df["N"] * df["D"]

    # Keep the max compute as the test point
    test_df_known = df.loc[df.N == df.N.max()]
    
    # Create an extrapolation limit as unknown test set
    test_df_unknown = pd.DataFrame.from_dict({
        "C": [test_df_known.C.max() * 10 ** i for i in range(1, 4)],
        "N": [None] * 3,
        "D": [None] * 3,
        "lr": [None] * 3,
        "gbsz": [None] * 3,
        "loss": [None] * 3,
    })

    train_df = df.loc[df.N != df.N.max()]
    
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
) -> tuple[np.ndarray, float]:
    if model_type in MODELS:
        model = MODELS[model_type]()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(train_x, train_y)
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
    if fit_quality == 'debug':
        initial_grid = [
            [l0, a, b]
            for l0 in np.log([0.5, 1.0, 2.0])     # L0 guesses
            for a in [0.0, 3.0, 6.0]              # a = log(coefficient)
            for b in [-0.03, -0.1, -0.3]          # exponent (negative!)
        ]
    elif fit_quality == 'dense':
        initial_grid = [
            [l0, a, b]
            for l0 in np.log([0.5, 0.75, 1.0, 1.5, 2.0])  # L0 guesses
            for a in [0.0, 1.5, 3.0, 4.5, 6.0]            # a = log(coefficient)
            for b in [-0.03, -0.06, -0.1, -0.2, -0.3]     # exponent (negative!)
        ]
    else:
        raise ValueError(f"Unsupported fit quality: {fit_quality}")
    bounds = [
        (np.log(1e-3), np.log(5.0)),  # L0 bounds
        (None, None),                 # A bounds
        (-1.0, 0.0)                   # alpha bounds
    ]
    return initial_grid, bounds


def _print_fit_results(params: np.ndarray):
    assert len(params) == 3, "Expected 3 parameters: L0, A, alpha"
    L0, A, alpha = params
    print(f"L(C) = {np.exp(L0):.3f} + {np.exp(A):.3f} * C^({alpha:.4f})\n")
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--niters', type=int, default=None)
    parser.add_argument('--total_C', type=float, default=None)
    parser.add_argument('--init_points', type=int, default=25)
    parser.add_argument('--init_compute_factor', type=float, default=10.0)
    parser.add_argument('--model_type', type=str, default='gp', choices=['gp'])
    parser.add_argument('--selection', type=str, default='random', choices=list(SELECTION))
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fit_quality', type=str, default='debug', choices=['debug', 'dense'])
    parser.add_argument('--C_envelope_growth', type=float, default=2.0)
    parser.add_argument('--filename', default="history")

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
    train_df, test_df_known, test_df_unknown = data_preprocess(df)

    # ground truth prep
    _global_df = pd.concat((train_df, test_df_known))
    print("\nBuilding global surrogate on all data as a baseline...")
    global_pred, global_mse = build_surrogate(
        np.log(_global_df[INPUT_X].to_numpy()),
        _global_df[OUTPUT_Y].to_numpy()[:, None],
        np.log(test_df_known[INPUT_X].to_numpy()),
        model_type=args.model_type
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

    # initialization: seed with the cheapest point, then fill up to init_points under the envelope
    C_lim = train_df.C.min() * args.init_compute_factor
    acquired_idx = pd.Index([train_df["C"].idxmin()])
    while len(acquired_idx) < args.init_points:
        pool = next_batch_compute(train_df, acquired_idx, C_lim)
        acquired_idx = acquired_idx.append(pd.Index([splitter.next_point(train_df, pool.index)]))
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
        pool = next_batch_compute(train_df, acquired_idx, C_lim)
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
        )

        # true loss where observed, predicted loss elsewhere
        df_pred = train_df.copy()
        df_pred.loc[remaining_idx, OUTPUT_Y] = pred_y

        # parametric fit on the mixed data
        best_params, best_loss = build_fit(df_pred[INPUT_X + ["C", OUTPUT_Y]], fit_quality=args.fit_quality)
        print(f"Iteration {iter_idx}: Fitting loss={best_loss:2.2e}", end="; ")
        _print_fit_results(best_params)

        history.append({
            "iter": iter_idx,
            "n_acquired": len(acquired_idx),
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

        # update C_lim envelope for sampling: ceiling stays one growth step above the max cost seen
        C_lim = max(C_lim, observations.C.max() * args.C_envelope_growth)
    
    print("Global surrogate fit results:")
    _print_fit_results(global_best_params)
    print(f"Known Test MSE: {history[-1]['known_test_mse']:.2e}")
    print(f"Unknown Test Predictions: {history[-1]['L_unknown']}")
    print("Run done!")

    print(f"{'='*30}\nIncumbent regrets:")
    print(f"L0: {incumbents['l0']:.3f} (signed)\nA: {incumbents['A']:.3f}\nalpha: {incumbents['alpha']:.4f}\n{'='*30}")

    # saving to disk
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_parquet(output_dir / f"{args.filename}.parquet", index=False)
    print(f"Saved history to {output_dir / f'{args.filename}.parquet'}")
# end of file