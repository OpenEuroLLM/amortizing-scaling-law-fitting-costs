"""L0 form: L(C) = L0 + A * C^alpha."""
import numpy as np

from scaling.utils import functional_form_L0

from .base import ParametricForm


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
    target="loss",
    use_pareto=True,
    initial_grid=_l0_initial_grid,
    bounds=[
        (1e-3, 4.0),  # L0 bounds
        (None, None),                 # A bounds
        (-1.0, 0.0),                  # alpha bounds
    ],
    interpretable=_l0_interpretable,
    signed_params=frozenset({"L0"}),
    to_L_vs_C=_l0_to_L_vs_C,
    format_fit=_l0_format,
)
