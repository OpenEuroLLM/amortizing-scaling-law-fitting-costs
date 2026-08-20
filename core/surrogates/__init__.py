"""Surrogate models that infill the loss grid.

``MODELS`` maps CLI names to surrogate classes. ``build_surrogate`` is the thin
fit-then-predict helper the acquisition loop calls each iteration.
"""

import numpy as np

from .base import SurrogateModel, NullSurrogate
from .gp import GPModel
from .tabpfn import TabPFNEnsembleModel, TabPFNBOModel
from .autogluon import AutoGluonModel
from .xgboost import XGBoostEnsembleModel

# Only ensemble variants are exposed for TabPFN / XGBoost -- a single tree model or
# single TabPFN call gives no usable predictive uncertainty. The one single-model
# exception is `tabpfn_bo`, which derives a genuine posterior from TabPFN's bar
# distribution (see TabPFNBOModel) and so is a valid Bayesian-optimization surrogate.
MODELS = {
    "none": NullSurrogate,
    "gp": GPModel,
    "tabpfn_ens": TabPFNEnsembleModel,
    "tabpfn_bo": TabPFNBOModel,
    "ag": AutoGluonModel,
    "xgb_ens": XGBoostEnsembleModel,
}


def build_surrogate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray | None = None,
    model_type: str = "gp",
    **kwargs
) -> tuple[np.ndarray, float, object]:
    if model_type in MODELS:
        model = MODELS[model_type](**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(train_x, train_y, **kwargs)
    pred_y = model.predict(test_x)
    mse = None
    if test_y is not None:
        mse = np.mean((test_y - pred_y.flatten()) ** 2)

    return pred_y, mse, model


__all__ = [
    "SurrogateModel",
    "NullSurrogate",
    "GPModel",
    "TabPFNEnsembleModel",
    "TabPFNBOModel",
    "AutoGluonModel",
    "XGBoostEnsembleModel",
    "MODELS",
    "build_surrogate",
]
