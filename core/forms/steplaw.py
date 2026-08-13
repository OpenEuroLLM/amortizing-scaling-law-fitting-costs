"""Step Law optimal LR / batch size (Wang et al. 2025, arXiv:2503.04715).

These predict *hyperparameters*, not loss: eta_opt(N,D) = A*N^p*D^q, B_opt(D) = A'*D^r.
They plug into ``build_fit``/``fit_parametric_form`` like any other form (their ``target``
is 'lr'/'gbsz' instead of 'loss'), but they have no meaningful L-vs-C marginalization, so
``to_L_vs_C`` raises rather than silently returning loss-shaped nonsense -- don't drive
these through ``main.py``'s C-extrapolation path.
"""
import numpy as np

from scaling.utils import functional_form_steplaw_lr, functional_form_steplaw_bsz

from .base import ParametricForm


def _steplaw_no_L_vs_C(C_grid: np.ndarray, params: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "steplaw forms predict optimal LR/batch size directly from (N, D), not loss vs "
        "compute -- there's no well-defined L-vs-C marginalization. Fit and evaluate via "
        "`.func`/`build_fit(..., y_name=...)` directly instead of the CLI's C-extrapolation "
        "panels."
    )


# ------------------------------------------------------------------
# Step Law optimal learning rate: eta(N, D) = A * N^p * D^q
# ------------------------------------------------------------------

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
    target="lr",
    use_pareto=False,
    initial_grid=_steplaw_lr_initial_grid,
    bounds=[(None, None), (None, None), (None, None)],
    interpretable=_steplaw_lr_interpretable,
    signed_params=frozenset(),
    to_L_vs_C=_steplaw_no_L_vs_C,
    format_fit=_steplaw_lr_format,
)


# ------------------------------------------------------------------
# Step Law optimal batch size: B(D) = A * D^r
# ------------------------------------------------------------------

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
    target="gbsz",
    use_pareto=False,
    initial_grid=_steplaw_bsz_initial_grid,
    bounds=[(None, None), (None, None)],
    interpretable=_steplaw_bsz_interpretable,
    signed_params=frozenset(),
    to_L_vs_C=_steplaw_no_L_vs_C,
    format_fit=_steplaw_bsz_format,
)
