from typing import List
import numpy as np
import torch
import gpytorch
from gpytorch.constraints import Interval


class _ConstrainedMean(gpytorch.means.Mean):
    """Base class registering parameters with optional box constraints.

    Constraints come from the same `param_fit_setup_args()` used by the Huber
    fits, so the GP mean is optimized within the bounds we consider plausible.
    gpytorch stores the unconstrained `raw_<name>` and exposes `<name>` as the
    transformed (in-bounds) value.
    """

    PARAM_NAMES: tuple[str, ...] = ()

    def _register(self, name: str, value: float, bound: tuple | None):
        raw_name = f"raw_{name}"
        self.register_parameter(
            raw_name, torch.nn.Parameter(torch.tensor(float(value), dtype=torch.float64))
        )
        if bound is not None and bound[0] is not None and bound[1] is not None:
            constraint = Interval(
                torch.tensor(float(bound[0]), dtype=torch.float64),
                torch.tensor(float(bound[1]), dtype=torch.float64),
            )
            self.register_constraint(raw_name, constraint)
            # set the raw value such that the constrained value == `value`
            with torch.no_grad():
                getattr(self, raw_name).copy_(
                    constraint.inverse_transform(
                        torch.tensor(float(np.clip(value, *bound)), dtype=torch.float64)
                    )
                )

    def _value(self, name: str) -> torch.Tensor:
        """Constrained value of a registered parameter."""
        raw = getattr(self, f"raw_{name}")
        constraint = getattr(self, f"raw_{name}_constraint", None)
        return raw if constraint is None else constraint.transform(raw)

    def named_values(self) -> dict:
        """Constrained parameter values, for diagnostics."""
        return {n: self._value(n).item() for n in self.PARAM_NAMES}


class ScalingLawMean_L0_form(_ConstrainedMean):
    """L(C) = L0 + A * C^b, parameterized as [l0, a, b] with L0 = exp(l0), A = exp(a)."""

    PARAM_NAMES = ("l0", "a", "b")

    def __init__(
        self,
        init_params: List[float] = (0.2, 3.5, -0.5),
        col_N: int = 0,
        col_D: int = 1,
        bounds: List[tuple] | None = None,
        use_bounds: bool = True,
        **kwargs
    ):
        super().__init__()  # log L0, log A, b
        if bounds is None and use_bounds:
            from parametric_forms import PARAMETRIC_FORMS
            bounds = PARAMETRIC_FORMS["L0"].bounds
        bounds = bounds or [None] * len(self.PARAM_NAMES)
        # registering to ensure they are optimized during MLL training
        for name, value, bound in zip(self.PARAM_NAMES, init_params, bounds):
            self._register(name, value, bound)
        self.col_N = col_N
        self.col_D = col_D

    def forward(self, x):
        logC = torch.log(torch.tensor(6.0, dtype=torch.float64)) + x[..., self.col_N] + x[..., self.col_D]        # cols 0,1 = log N, log D
        return self._value("l0").exp() + torch.exp(self._value("a") + self._value("b") * logC)


class ScalingLawMean_chin3_form(_ConstrainedMean):
    """Chinchilla Approach-3 mean: L(N, D) = A/N^alpha + B/D^beta + E.

    Stable parameterization [a, alpha, b, beta, e] with A = exp(a), B = exp(b),
    E = exp(e), so L = exp(a - alpha*logN) + exp(b - beta*logD) + exp(e).
    Inputs x are log-features; cols col_N, col_D hold log N, log D.
    """

    PARAM_NAMES = ("a", "alpha", "b", "beta", "e")

    def __init__(
        self,
        init_params: List[float] = (6.0, 0.34, 6.0, 0.28, 0.5),  # ~Chinchilla fit
        col_N: int = 0,
        col_D: int = 1,
        bounds: List[tuple] | None = None,
        use_bounds: bool = True,
        **kwargs
    ):
        super().__init__()  # log A, alpha, log B, beta, log E
        if bounds is None and use_bounds:
            from parametric_forms import PARAMETRIC_FORMS
            bounds = PARAMETRIC_FORMS["chinchilla3"].bounds
        bounds = bounds or [None] * len(self.PARAM_NAMES)
        # registering to ensure they are optimized during MLL training
        for name, value, bound in zip(self.PARAM_NAMES, init_params, bounds):
            self._register(name, value, bound)
        self.col_N = col_N
        self.col_D = col_D

    def forward(self, x):
        logN = x[..., self.col_N]
        logD = x[..., self.col_D]
        return (
            torch.exp(self._value("a") - self._value("alpha") * logN)
            + torch.exp(self._value("b") - self._value("beta") * logD)
            + self._value("e").exp()
        )


# Keys mirror `parametric_forms.PARAMETRIC_FORMS` so --form and --gp_mean_form share a
# vocabulary. Forms without a torch mean implementation simply have no entry here.
MEAN_FORMS: dict[str, type] = {
    "L0": ScalingLawMean_L0_form,
    "chinchilla3": ScalingLawMean_chin3_form,
}
# end of file