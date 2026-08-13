"""Width/depth form: L(width, depth, D) = E + A/width^alpha + B/depth^beta + F/D^gamma.

Chinchilla Approach 3 extended with a third independent additive term for D, fit against
the ``width``/``depth``/``D`` columns. Reuses the 7-parameter interaction grid.

NOTE on to_L_vs_C: there's no standard compute proxy relating width, depth, and D jointly
(D is already the compute-relevant axis, so a width*depth*D proxy would double-count it),
so there's no well-defined Lagrangian split -- this raises like the steplaw forms.
"""
import numpy as np

from scaling.utils import functional_form_width_depth

from .base import ParametricForm
from .chinchilla import _chin_interaction_initial_grid


def _width_depth_interpretable(params: np.ndarray) -> dict[str, float]:
    a, alpha, b, beta, f, gamma, e = params
    return {
        "A": float(np.exp(a)), "alpha_width": float(alpha),
        "B": float(np.exp(b)), "beta_depth": float(beta),
        "F": float(np.exp(f)), "gamma_D": float(gamma),
        "E": float(np.exp(e)),
    }


def _width_depth_format(params: np.ndarray) -> str:
    interp = _width_depth_interpretable(params)
    return (
        f"L(width,depth,D) = {interp['E']:.3f} + {interp['A']:.3f}/width^{interp['alpha_width']:.4f} "
        f"+ {interp['B']:.3f}/depth^{interp['beta_depth']:.4f} "
        f"+ {interp['F']:.3f}/D^{interp['gamma_D']:.4f}"
    )


def _width_depth_no_L_vs_C(C_grid: np.ndarray, params: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "width_depth has no standard compute proxy relating width, depth, and D jointly "
        "(D is already the compute-relevant axis, so width*depth*D would double-count it) -- "
        "there's no well-defined L-vs-C marginalization. Fit and evaluate via `.func`/"
        "`build_fit(..., y_name=...)` directly instead of the CLI's C-extrapolation panels."
    )


WIDTH_DEPTH_FORM = ParametricForm(
    name="width_depth",
    func=functional_form_width_depth,
    input_cols=["width", "depth", "D"],
    target="loss",
    use_pareto=False,
    initial_grid=_chin_interaction_initial_grid,
    bounds=[
        (None, None),                 # A bounds (log-space)
        (1e-3, None),                 # alpha_width bounds (must stay positive)
        (None, None),                 # B bounds (log-space)
        (1e-3, None),                 # beta_depth bounds (must stay positive)
        (None, None),                 # F bounds (log-space)
        (1e-3, None),                 # gamma_D bounds (must stay positive)
        (np.log(1e-3), np.log(4.0)),  # E (irreducible loss) bounds
    ],
    interpretable=_width_depth_interpretable,
    signed_params=frozenset({"E"}),
    to_L_vs_C=_width_depth_no_L_vs_C,
    format_fit=_width_depth_format,
)
