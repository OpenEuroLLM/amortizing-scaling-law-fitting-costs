"""Registry of parametric L(...) forms usable by the fitting pipeline.

Each `ParametricForm` bundles everything `iterative_fits_L0_form.py` needs to fit and
report on a functional form: the underlying function (from the `scaling.utils`
submodule), its optimizer-space parameterization, an initial-grid/bounds preset per
`fit_quality`, a mapping to human-readable parameter values, and a way to render the
fit as L vs C (for forms not natively indexed by C, this marginalizes via the
form's own compute-optimal allocation).

To add a new form: write the two callables below (mirroring `_l0_*` or `_chin3_*`),
wrap them in a `ParametricForm`, and add it to `PARAMETRIC_FORMS`. Nothing else in the
fitting pipeline needs to change.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from scaling.utils import (
    functional_form_L0,
    functional_form_chin3,
    functional_form_chin_interaction,
    functional_form_steplaw_lr,
    functional_form_steplaw_bsz,
)


@dataclass
class ParametricForm:
    name: str
    func: Callable                                       # (data, params, return_log_loss=True) -> L or log L
    input_cols: list[str]                                 # df columns fed as `data` to func, in order
    use_pareto: bool                                       # restrict to the loss envelope before fitting
    initial_grid: Callable[[str], list[list[float]]]        # fit_quality -> list of optimizer-space init points
    bounds: list[tuple]                                    # bounds on optimizer-space params, same order as init
    interpretable: Callable[[np.ndarray], dict[str, float]]  # optimizer-space params -> human-readable values
    signed_params: frozenset                                # interpretable keys whose regret keeps sign (irreducible-loss-like terms)
    to_L_vs_C: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (C_grid, params) -> L, for cross-form diagnostics/plots
    format_fit: Callable[[np.ndarray], str]                 # optimizer-space params -> human-readable equation string


# ============================================================
# L0 form: L(C) = L0 + A * C^alpha
# ============================================================

def _l0_initial_grid(fit_quality: str) -> list[list[float]]:
    if fit_quality == "debug":  # 3 x 3 x 3 = 27
        return [
            [l0, a, b]
            for l0 in np.log([0.5, 1.0, 2.0])   # L0 guesses
            for a in [0., 5., 10.]              # a = log(coefficient)
            for b in [-0.05, -0.2, -0.5]        # exponent (negative!)
        ]
    if fit_quality == "dense":  # 6 x 6 x 6 = 216
        return [
            [l0, a, b]
            for l0 in np.linspace(np.log(0.1), np.log(3.0), 6)
            for a in [0., 5., 10., 20., 30., 40.]
            for b in [-0.02, -0.05, -0.1, -0.2, -0.5, -1.0]
        ]
    if fit_quality == "super_dense":  # 8 x 10 x 10 = 800
        return [
            [l0, a, b]
            for l0 in np.linspace(np.log(0.05), np.log(3.5), 8)
            for a in [0., 2.5, 5., 7.5, 10., 15., 20., 25., 35., 45.]
            for b in [-0.01, -0.02, -0.05, -0.08, -0.12, -0.2, -0.35, -0.5, -0.75, -1.0]
        ]
    raise ValueError(f"Unsupported fit quality: {fit_quality}")


def _l0_interpretable(params: np.ndarray) -> dict[str, float]:
    l0, a, b = params
    return {"L0": float(np.exp(l0)), "A": float(np.exp(a)), "alpha": float(b)}


def _l0_format(params: np.ndarray) -> str:
    interp = _l0_interpretable(params)
    return f"L(C) = {interp['L0']:.3f} + {interp['A']:.3f} * C^({interp['alpha']:.4f})"


def _l0_to_L_vs_C(C_grid: np.ndarray, params: np.ndarray) -> np.ndarray:
    return functional_form_L0(np.asarray(C_grid, dtype=float), params, return_log_loss=False)


L0_FORM = ParametricForm(
    name="L0",
    func=functional_form_L0,
    input_cols=["C"],
    use_pareto=True,
    initial_grid=_l0_initial_grid,
    bounds=[
        (np.log(1e-3), np.log(4.0)),  # L0 bounds
        (None, None),                 # A bounds
        (-1.0, 0.0),                  # alpha bounds
    ],
    interpretable=_l0_interpretable,
    signed_params=frozenset({"L0"}),
    to_L_vs_C=_l0_to_L_vs_C,
    format_fit=_l0_format,
)


# ============================================================
# Chinchilla Approach 3: L(N, D) = E + A / N^alpha + B / D^beta
# ============================================================
# functional_form_chin3 expects params = [a, alpha, b, beta, e], with a=log(A), b=log(B), e=log(E).
#
# use_pareto=False: unlike the L0 form, this needs N and D as independently-varying axes to
# identify alpha and beta separately. Restricting to the compute-optimal (Pareto) envelope of
# L vs C would make N and D nearly collinear there, so the fit is done on the full dataset.

def _chin_grid_values(fit_quality: str) -> tuple[np.ndarray, list[float], list[float]]:
    """Shared (e_vals, coef_vals, exp_vals) grid presets for chinchilla-style N/D power terms.

    Reused by both the plain chinchilla3 grid and the interaction-term grid below, so the
    two forms stay consistent as the presets are tuned.
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


# ============================================================
# Chinchilla + N*D interaction: L(N, D) = E + A/N^alpha + B/D^beta + G/(N*D)^gamma
# ============================================================
# functional_form_chin_interaction expects params = [a, alpha, b, beta, g, gamma, e].
#
# use_pareto=False for the same reason as chinchilla3: N and D need to vary independently
# to identify alpha, beta, gamma separately.
#
# Compute-optimal (N, D): along the constraint N*D = C/6, the interaction term (N*D)^-gamma
# is a constant (it only depends on the product, not the split), so it drops out of the
# Lagrangian argmin entirely -- the optimal split is identical to plain chinchilla3's, and
# `chinchilla_compute_optimal_ND` is reused as-is. Only the *value* of L at that split picks
# up the extra interaction term.

def _chin_interaction_initial_grid(fit_quality: str) -> list[list[float]]:
    # debug: 2x2x2x2x2x2x2=128, dense: 3^7=2187, super_dense: 4^7=16384
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


# ============================================================
# Width/depth form: L(width, depth) = E + A/width^alpha + B/depth^beta
# ============================================================
# Structurally identical to chinchilla3 (reuses functional_form_chin3 directly), just
# fit against `width`/`depth` columns instead of `N`/`D`.
#
# NOTE on to_L_vs_C: there's no universally agreed compute formula for (width, depth) the
# way C = 6*N*D is standard for (N, D). This reuses `chinchilla_compute_optimal_ND`'s closed-
# form Lagrangian split under that same C = 6*width*depth proxy. If your data's compute
# accounting for width/depth differs, adjust `_width_depth_to_L_vs_C` accordingly.

def _width_depth_interpretable(params: np.ndarray) -> dict[str, float]:
    a, alpha, b, beta, e = params
    return {
        "A": float(np.exp(a)), "alpha_width": float(alpha),
        "B": float(np.exp(b)), "beta_depth": float(beta),
        "E": float(np.exp(e)),
    }


def _width_depth_format(params: np.ndarray) -> str:
    interp = _width_depth_interpretable(params)
    return (
        f"L(width,depth) = {interp['E']:.3f} + {interp['A']:.3f}/width^{interp['alpha_width']:.4f} "
        f"+ {interp['B']:.3f}/depth^{interp['beta_depth']:.4f}"
    )


def _width_depth_to_L_vs_C(C_grid: np.ndarray, params: np.ndarray) -> np.ndarray:
    interp = _width_depth_interpretable(params)
    width_opt, depth_opt = chinchilla_compute_optimal_ND(
        np.asarray(C_grid, dtype=float), interp["A"], interp["alpha_width"], interp["B"], interp["beta_depth"]
    )
    data = np.stack([width_opt, depth_opt], axis=-1)
    return functional_form_chin3(data, params, return_log_loss=False)


WIDTH_DEPTH_FORM = ParametricForm(
    name="width_depth",
    func=functional_form_chin3,
    input_cols=["width", "depth"],
    use_pareto=False,
    initial_grid=_chin3_initial_grid,
    bounds=[
        (None, None),                 # A bounds (log-space)
        (1e-3, None),                 # alpha_width bounds (must stay positive)
        (None, None),                 # B bounds (log-space)
        (1e-3, None),                 # beta_depth bounds (must stay positive)
        (np.log(1e-3), np.log(4.0)),  # E (irreducible loss) bounds
    ],
    interpretable=_width_depth_interpretable,
    signed_params=frozenset({"E"}),
    to_L_vs_C=_width_depth_to_L_vs_C,
    format_fit=_width_depth_format,
)


# ============================================================
# Step Law optimal LR / batch size (Wang et al. 2025, arXiv:2503.04715)
# ============================================================
# These predict *hyperparameters*, not loss: eta_opt(N,D) = A*N^p*D^q, B_opt(D) = A'*D^r.
# They plug into `build_fit`/`fit_parametric_form` like any other form (fit with
# `y_name='lr'` or `y_name='gbsz'` instead of the default 'loss'), but they have no
# meaningful L-vs-C marginalization, so `to_L_vs_C` raises rather than silently returning
# loss-shaped nonsense -- don't drive these through `iterative_fits_L0_form.py`'s
# C-extrapolation path.

def _steplaw_no_L_vs_C(C_grid: np.ndarray, params: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "steplaw forms predict optimal LR/batch size directly from (N, D), not loss vs "
        "compute -- there's no well-defined L-vs-C marginalization. Fit and evaluate via "
        "`.func`/`build_fit(..., y_name=...)` directly instead of the CLI's C-extrapolation "
        "panels."
    )


def _steplaw_lr_initial_grid(fit_quality: str) -> list[list[float]]:
    if fit_quality == "debug":  # 3 x 3 x 3 = 27
        a_vals = np.log([0.1, 1.0, 10.0])
        p_vals = [-1.0, -0.5, 0.0]
        q_vals = [0.0, 0.3, 0.6]
    elif fit_quality == "dense":  # 5 x 5 x 5 = 125
        a_vals = np.linspace(np.log(0.01), np.log(50.0), 5)
        p_vals = np.linspace(-1.2, 0.2, 5)
        q_vals = np.linspace(-0.2, 1.0, 5)
    elif fit_quality == "super_dense":  # 7 x 7 x 7 = 343
        a_vals = np.linspace(np.log(0.005), np.log(100.0), 7)
        p_vals = np.linspace(-1.5, 0.3, 7)
        q_vals = np.linspace(-0.3, 1.2, 7)
    else:
        raise ValueError(f"Unsupported fit quality: {fit_quality}")
    return [[a, p, q] for a in a_vals for p in p_vals for q in q_vals]


def _steplaw_lr_interpretable(params: np.ndarray) -> dict[str, float]:
    a, p, q = params
    return {"coef": float(np.exp(a)), "N_exponent": float(p), "D_exponent": float(q)}


def _steplaw_lr_format(params: np.ndarray) -> str:
    interp = _steplaw_lr_interpretable(params)
    return (
        f"eta(N,D) = {interp['coef']:.4g} * N^{interp['N_exponent']:.4f} "
        f"* D^{interp['D_exponent']:.4f}"
    )


STEPLAW_LR_FORM = ParametricForm(
    name="steplaw_lr",
    func=functional_form_steplaw_lr,
    input_cols=["N", "D"],
    use_pareto=False,
    initial_grid=_steplaw_lr_initial_grid,
    bounds=[(None, None), (None, None), (None, None)],
    interpretable=_steplaw_lr_interpretable,
    signed_params=frozenset(),
    to_L_vs_C=_steplaw_no_L_vs_C,
    format_fit=_steplaw_lr_format,
)


def _steplaw_bsz_initial_grid(fit_quality: str) -> list[list[float]]:
    if fit_quality == "debug":  # 3 x 3 = 9
        a_vals = np.log([1.0, 32.0, 256.0])
        r_vals = [0.0, 0.3, 0.6]
    elif fit_quality == "dense":  # 6 x 6 = 36
        a_vals = np.linspace(np.log(0.5), np.log(1024.0), 6)
        r_vals = np.linspace(-0.2, 1.0, 6)
    elif fit_quality == "super_dense":  # 8 x 8 = 64
        a_vals = np.linspace(np.log(0.1), np.log(4096.0), 8)
        r_vals = np.linspace(-0.3, 1.2, 8)
    else:
        raise ValueError(f"Unsupported fit quality: {fit_quality}")
    return [[a, r] for a in a_vals for r in r_vals]


def _steplaw_bsz_interpretable(params: np.ndarray) -> dict[str, float]:
    a, r = params
    return {"coef": float(np.exp(a)), "D_exponent": float(r)}


def _steplaw_bsz_format(params: np.ndarray) -> str:
    interp = _steplaw_bsz_interpretable(params)
    return f"B(D) = {interp['coef']:.4g} * D^{interp['D_exponent']:.4f}"


STEPLAW_BSZ_FORM = ParametricForm(
    name="steplaw_bsz",
    func=functional_form_steplaw_bsz,
    input_cols=["D"],
    use_pareto=False,
    initial_grid=_steplaw_bsz_initial_grid,
    bounds=[(None, None), (None, None)],
    interpretable=_steplaw_bsz_interpretable,
    signed_params=frozenset(),
    to_L_vs_C=_steplaw_no_L_vs_C,
    format_fit=_steplaw_bsz_format,
)


PARAMETRIC_FORMS: dict[str, ParametricForm] = {
    "L0": L0_FORM,
    "chinchilla3": CHINCHILLA3_FORM,
    "chinchilla_interaction": CHINCHILLA_INTERACTION_FORM,
    "width_depth": WIDTH_DEPTH_FORM,
    "steplaw_lr": STEPLAW_LR_FORM,
    "steplaw_bsz": STEPLAW_BSZ_FORM,
}
# end of file
