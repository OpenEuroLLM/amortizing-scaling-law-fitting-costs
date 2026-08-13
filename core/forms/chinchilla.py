"""Chinchilla Approach 3 and its N*D-interaction extension.

- chinchilla3:            L(N, D) = E + A/N^alpha + B/D^beta
- chinchilla_interaction: chinchilla3 + G/(N*D)^gamma

Both use ``use_pareto=False``: unlike the L0 form, they need N and D as independently
varying axes to identify the exponents separately. Restricting to the compute-optimal
(Pareto) envelope of L vs C would make N and D nearly collinear there.
"""
import numpy as np

from scaling.utils import functional_form_chin3, functional_form_chin_interaction

from .base import ParametricForm


def _chin_grid_values(fit_quality: str) -> tuple[np.ndarray, list[float], list[float]]:
    """Shared (e_vals, coef_vals, exp_vals) grid presets for chinchilla-style N/D power terms.

    Reused by the plain chinchilla3 grid, the interaction-term grid, and the width/depth
    form, so they stay consistent as the presets are tuned.
    """
    if fit_quality == "debug":  # 2 x 2 x 2 per term
        e_vals = np.log([0.5, 2.0])
        coef_vals = [0., 10.]      # log(A), log(B), ...
        exp_vals = [0.1, 0.5]      # alpha, beta, ...
    elif fit_quality == "dense":  # 3 x 3 x 3 per term
        e_vals = np.linspace(np.log(0.1), np.log(3.0), 3)
        coef_vals = [0., 10., 20.]
        exp_vals = [0.05, 0.2, 0.4]
    elif fit_quality == "super_dense":  # 4 x 4 x 4 per term
        e_vals = np.linspace(np.log(0.05), np.log(3.5), 4)
        coef_vals = [0., 10., 20., 35.]
        exp_vals = [0.02, 0.1, 0.25, 0.45]
    else:
        raise ValueError(f"Unsupported fit quality: {fit_quality}")
    return e_vals, coef_vals, exp_vals


def chinchilla_compute_optimal_ND(
    C: np.ndarray, A: float, alpha: float, B: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute-optimal (N, D) minimizing A*N^-alpha + B*D^-beta subject to C = 6*N*D.

    Closed form via Lagrange multipliers (Hoffmann et al. 2022, Approach 3):
    N_opt(C) = G * (C/6)^(beta/(alpha+beta)), G = ((alpha*A)/(beta*B))^(1/(alpha+beta)).
    """
    G = ((alpha * A) / (beta * B)) ** (1.0 / (alpha + beta))
    N_opt = G * (C / 6.0) ** (beta / (alpha + beta))
    D_opt = (C / 6.0) / N_opt
    return N_opt, D_opt


# ------------------------------------------------------------------
# chinchilla3
# functional_form_chin3 expects params = [a, alpha, b, beta, e], a=log(A), b=log(B), e=log(E).
# ------------------------------------------------------------------

def _chin3_initial_grid(fit_quality: str) -> list[list[float]]:
    # debug: 2x2x2x2x2=32, dense: 3x3x3x3x3=243, super_dense: 4x4x4x4x4=1024
    e_vals, coef_vals, exp_vals = _chin_grid_values(fit_quality)
    return [
        [a, alpha, b, beta, e]
        for e in e_vals
        for a in coef_vals
        for alpha in exp_vals
        for b in coef_vals
        for beta in exp_vals
    ]


def _chin3_interpretable(params: np.ndarray) -> dict[str, float]:
    a, alpha, b, beta, e = params
    return {
        "A": float(np.exp(a)), "alpha": float(alpha),
        "B": float(np.exp(b)), "beta": float(beta),
        "E": float(np.exp(e)),
    }


def _chin3_format(params: np.ndarray) -> str:
    interp = _chin3_interpretable(params)
    return (
        f"L(N,D) = {interp['E']:.3f} + {interp['A']:.3f}/N^{interp['alpha']:.4f} "
        f"+ {interp['B']:.3f}/D^{interp['beta']:.4f}"
    )


def _chin3_to_L_vs_C(C_grid: np.ndarray, params: np.ndarray) -> np.ndarray:
    interp = _chin3_interpretable(params)
    N_opt, D_opt = chinchilla_compute_optimal_ND(
        np.asarray(C_grid, dtype=float), interp["A"], interp["alpha"], interp["B"], interp["beta"]
    )
    data = np.stack([N_opt, D_opt], axis=-1)
    return functional_form_chin3(data, params, return_log_loss=False)


CHINCHILLA3_FORM = ParametricForm(
    name="chinchilla3",
    func=functional_form_chin3,
    input_cols=["N", "D"],
    target="loss",
    use_pareto=False,
    initial_grid=_chin3_initial_grid,
    bounds=[
        (None, None),                 # A bounds (log-space)
        (1e-3, None),                 # alpha bounds (must stay positive)
        (None, None),                 # B bounds (log-space)
        (1e-3, None),                 # beta bounds (must stay positive)
        (np.log(1e-3), np.log(4.0)),  # E (irreducible loss) bounds
    ],
    interpretable=_chin3_interpretable,
    signed_params=frozenset({"E"}),
    to_L_vs_C=_chin3_to_L_vs_C,
    format_fit=_chin3_format,
)


# ------------------------------------------------------------------
# chinchilla + N*D interaction
# functional_form_chin_interaction expects params = [a, alpha, b, beta, g, gamma, e].
#
# Compute-optimal (N, D): along N*D = C/6 the interaction term (N*D)^-gamma is constant
# (depends only on the product), so it drops out of the Lagrangian argmin -- the optimal
# split is identical to plain chinchilla3's, and `chinchilla_compute_optimal_ND` is reused.
# ------------------------------------------------------------------

def _chin_interaction_initial_grid(fit_quality: str) -> list[list[float]]:
    # debug: 2^7=128, dense: 3^7=2187, super_dense: 4^7=16384
    e_vals, coef_vals, exp_vals = _chin_grid_values(fit_quality)
    return [
        [a, alpha, b, beta, g, gamma, e]
        for e in e_vals
        for a in coef_vals
        for alpha in exp_vals
        for b in coef_vals
        for beta in exp_vals
        for g in coef_vals
        for gamma in exp_vals
    ]


def _chin_interaction_interpretable(params: np.ndarray) -> dict[str, float]:
    a, alpha, b, beta, g, gamma, e = params
    return {
        "A": float(np.exp(a)), "alpha": float(alpha),
        "B": float(np.exp(b)), "beta": float(beta),
        "G": float(np.exp(g)), "gamma": float(gamma),
        "E": float(np.exp(e)),
    }


def _chin_interaction_format(params: np.ndarray) -> str:
    interp = _chin_interaction_interpretable(params)
    return (
        f"L(N,D) = {interp['E']:.3f} + {interp['A']:.3f}/N^{interp['alpha']:.4f} "
        f"+ {interp['B']:.3f}/D^{interp['beta']:.4f} "
        f"+ {interp['G']:.3f}/(N*D)^{interp['gamma']:.4f}"
    )


def _chin_interaction_to_L_vs_C(C_grid: np.ndarray, params: np.ndarray) -> np.ndarray:
    interp = _chin_interaction_interpretable(params)
    N_opt, D_opt = chinchilla_compute_optimal_ND(
        np.asarray(C_grid, dtype=float), interp["A"], interp["alpha"], interp["B"], interp["beta"]
    )
    data = np.stack([N_opt, D_opt], axis=-1)
    return functional_form_chin_interaction(data, params, return_log_loss=False)


CHINCHILLA_INTERACTION_FORM = ParametricForm(
    name="chinchilla_interaction",
    func=functional_form_chin_interaction,
    input_cols=["N", "D"],
    target="loss",
    use_pareto=False,
    initial_grid=_chin_interaction_initial_grid,
    bounds=[
        (None, None),                 # A bounds (log-space)
        (1e-3, None),                 # alpha bounds (must stay positive)
        (None, None),                 # B bounds (log-space)
        (1e-3, None),                 # beta bounds (must stay positive)
        (None, None),                 # G bounds (log-space)
        (1e-3, None),                 # gamma bounds (must stay positive)
        (np.log(1e-3), np.log(4.0)),  # E (irreducible loss) bounds
    ],
    interpretable=_chin_interaction_interpretable,
    signed_params=frozenset({"E"}),
    to_L_vs_C=_chin_interaction_to_L_vs_C,
    format_fit=_chin_interaction_format,
)
