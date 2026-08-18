import numpy as np

from .base import SurrogateModel


# Default XGBRegressor hyperparameters, tuned for the small (tens-to-hundreds of
# points) log-loss regression the surrogate sees. Overridable via kwargs.
_XGB_DEFAULTS = dict(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=1,  # small-data surrogate: single-threaded is faster here and sidesteps
               # the macOS duplicate-OpenMP-runtime crash seen with multi-threaded xgboost
)


def _xgb_params(kwargs: dict) -> dict:
    params = dict(_XGB_DEFAULTS)
    params.update(kwargs.get("xgb_params", {}))
    return params


class XGBoostEnsembleModel(SurrogateModel):
    """Bagged ensemble of ``XGBRegressor``s over bootstrap resamples + seeds.

    Each member is trained on a bootstrap resample of the data with its own
    ``random_state``. ``predict`` returns the member mean, and with
    ``return_std=True`` the member std -- an uncertainty proxy that lets the
    ensemble drive variance-/EI-based selection.
    """

    def __init__(self, n_members: int = 16, random_state: int = 0, **kwargs):
        self.n_members = kwargs.get("xgb_n_members", n_members)
        self.random_state = random_state
        self._params = _xgb_params(kwargs)
        self._regs: list = []

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        from xgboost import XGBRegressor

        n_members = kwargs.get("xgb_n_members", self.n_members)
        train_y = train_y.ravel()
        n = len(train_x)
        self._regs = []
        for i in range(n_members):
            rng = np.random.default_rng(self.random_state + i)
            idx = rng.integers(0, n, size=n)  # bootstrap resample
            reg = XGBRegressor(random_state=self.random_state + i, **self._params)
            reg.fit(train_x[idx], train_y[idx])
            self._regs.append(reg)

    def predict(self, X_test: np.ndarray, return_std: bool = False):
        preds = np.stack([reg.predict(X_test).ravel() for reg in self._regs], axis=0)
        mean = preds.mean(axis=0)
        if return_std:
            return mean, preds.std(axis=0)
        return mean
