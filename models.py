from abc import ABC, abstractmethod

import numpy as np
from typing import List


# Names of the parametric GP means available in `mean_functions.MEAN_FORMS`, kept here
# so callers can offer the choice without importing torch/gpytorch. Keys mirror
# `parametric_forms.PARAMETRIC_FORMS`; validated against the real registry in fit().
MEAN_FORM_NAMES: tuple[str, ...] = ("L0", "chinchilla3")

# Kernel choices for GPModel; "matern" is the current default (Matern-5/2 with ARD).
KERNEL_NAMES: tuple[str, ...] = (
    "matern", "linear", "linear+matern",
    "linear_nd+matern_hp",   # additive: linear on (N,D) + Matern on hyperparams
)


class SurrogateModel(ABC):
    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """train_x and train_y are already log-transformed."""

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """X_test is already log-transformed. Returns 1-D predictions."""

    
class GPModel(SurrogateModel):
    def __init__(self, **kwargs):
        self._gp = None
        self._likelihood = None
        self.use_mean_model = False
        self._mean_kwargs = {}
        if "mean_kwargs" in kwargs:
            self._mean_kwargs.update(kwargs["mean_kwargs"])
        self.mean_form = None
        if "mean_form" in kwargs:
            self.mean_form = kwargs["mean_form"]
        self.kernel = kwargs.get("kernel", None) or "matern"

    def _build_kernel(self, n_dims: int):
        import gpytorch

        # Whole-input kernels (all dims share one kernel family, ARD per dim).
        if self.kernel == "matern":
            return gpytorch.kernels.MaternKernel(ard_num_dims=n_dims)
        if self.kernel == "linear":
            return gpytorch.kernels.LinearKernel(ard_num_dims=n_dims)
        if self.kernel == "linear+matern":
            return (
                gpytorch.kernels.LinearKernel(ard_num_dims=n_dims)
                + gpytorch.kernels.MaternKernel(ard_num_dims=n_dims)
            )

        # Grouped additive kernel: a linear kernel on the scale dims (N=0, D=1) plus a
        # Matern kernel on the remaining hyperparameter dims (2..n_dims-1). active_dims
        # scopes each kernel to its own columns.
        if self.kernel == "linear_nd+matern_hp":
            scale_dims = [0, 1]
            hp_dims = list(range(2, n_dims))
            if not hp_dims:
                raise ValueError(
                    f"kernel '{self.kernel}' needs >2 input dims (got n_dims={n_dims})."
                )
            k_scale = gpytorch.kernels.LinearKernel(
                ard_num_dims=len(scale_dims), active_dims=scale_dims
            )
            k_hp = gpytorch.kernels.MaternKernel(
                ard_num_dims=len(hp_dims), active_dims=hp_dims
            )
            return k_scale + k_hp

        raise ValueError(f"Unsupported kernel: {self.kernel}. Choose from {list(KERNEL_NAMES)}.")

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, max_attempts: int = 10, **kwargs) -> None:
        # lazy import to avoid requiring gpytorch/botorch unless GPModel is used
        import botorch, gpytorch, torch
        from mean_functions import MEAN_FORMS

        if self.mean_form is not None and self.mean_form not in MEAN_FORMS:
            raise ValueError(
                f"Unsupported mean_form: {self.mean_form}. Choose from {list(MEAN_FORMS)} or None."
            )

        self._gp = botorch.models.SingleTaskGP(
            torch.tensor(train_x),
            torch.tensor(train_y),
            mean_module=(
                MEAN_FORMS[self.mean_form](**self._mean_kwargs)
                if self.mean_form is not None else None
            ),
            outcome_transform=None,
            covar_module=self._build_kernel(train_x.shape[1]),
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._gp.likelihood, self._gp)
        botorch.fit_gpytorch_mll(mll, max_attempts=max_attempts)

        if kwargs.get("diagnose", False):
            with torch.no_grad():
                self._gp.train()
                mll_val = mll(self._gp(*self._gp.train_inputs), self._gp.train_targets).item()
                train_pred = self._gp.posterior(self._gp.train_inputs[0]).mean
                train_mse = ((train_pred - self._gp.train_targets.unsqueeze(-1)) ** 2).mean().item()
            _n = self._gp.train_targets.numel()
            print(f"  [GP] n={_n}  MLL/n={mll_val:.3f}  train_mse={train_mse:.2e}")
            print(f"  [GP] n={_n} noise={self._gp.likelihood.noise.item():.2e}")
            print(f"  [GP] n={_n} lengthscales={self._gp.covar_module.lengthscale.detach().numpy().ravel().round(2)}")  # N,D,lr,gbsz
            _mean = self._gp.mean_module
            _mean_vals = (
                _mean.named_values() if hasattr(_mean, "named_values")
                else {n: p.item() for n, p in _mean.named_parameters()}
            )
            print(f"  [GP] n={_n} mean={ {k: round(v, 3) for k, v in _mean_vals.items()} }")

        self._gp.eval()


    def predict(self, X_test: np.ndarray, return_std: bool = False):
        import torch

        with torch.no_grad():
            post = self._gp.posterior(torch.tensor(X_test))
            mean = post.mean.numpy().squeeze(-1)
            if return_std:
                std = post.variance.clamp_min(1e-12).sqrt().numpy().squeeze(-1)
                return mean, std
            return mean


class TabPFNModel(SurrogateModel):
    def fit(self, train_x: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        from tabpfn import TabPFNRegressor

        self._reg = TabPFNRegressor()
        self._reg.fit(train_x, train_y)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._reg.predict(X_test)


class AutoGluonModel(SurrogateModel):
    def fit(self, train_x: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        import pandas as pd
        from autogluon.tabular import TabularPredictor

        train_data = pd.DataFrame(train_x, columns=[f"f{i}" for i in range(train_x.shape[1])])
        train_data["labels"] = train_y.ravel()
        self._predictor = TabularPredictor(label="labels").fit(train_data)
        self._feature_names = train_data.columns.drop("labels").tolist()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        import pandas as pd

        X_df = pd.DataFrame(X_test, columns=self._feature_names)
        return self._predictor.predict(X_df).to_numpy()


MODELS = {
    "gp": GPModel,
    "tabpfn": TabPFNModel,
    "ag": AutoGluonModel,
}
