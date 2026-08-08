from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ParametricForm:
    """Everything the fitting pipeline needs to fit and report on a functional form.

    Bundles the underlying function (from the ``scaling.utils`` submodule), its
    optimizer-space parameterization, an initial-grid/bounds preset per ``fit_quality``,
    a mapping to human-readable parameter values, and a way to render the fit as L vs C
    (for forms not natively indexed by C, this marginalizes via the form's own
    compute-optimal allocation).

    To add a new form: write the callables (mirroring ``core.forms.l0`` or
    ``core.forms.chinchilla``), wrap them in a ``ParametricForm``, and register it in
    ``core.forms.PARAMETRIC_FORMS``. Nothing else in the fitting pipeline needs to change.
    """
    name: str
    func: Callable                                       # (data, params, return_log_loss=True) -> L or log L
    input_cols: list[str]                                 # df columns fed as `data` to func, in order
    target: str                                           # df column build_fit() fits against by default
    use_pareto: bool                                       # restrict to the loss envelope before fitting
    initial_grid: Callable[[str], list[list[float]]]        # fit_quality -> list of optimizer-space init points
    bounds: list[tuple]                                    # bounds on optimizer-space params, same order as init
    interpretable: Callable[[np.ndarray], dict[str, float]]  # optimizer-space params -> human-readable values
    signed_params: frozenset                                # interpretable keys whose regret keeps sign (irreducible-loss-like terms)
    to_L_vs_C: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (C_grid, params) -> L, for cross-form diagnostics/plots
    format_fit: Callable[[np.ndarray], str]                 # optimizer-space params -> human-readable equation string
