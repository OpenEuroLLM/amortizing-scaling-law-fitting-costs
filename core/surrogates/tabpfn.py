import numpy as np

from .base import SurrogateModel


class TabPFNModel(SurrogateModel):
    def __init__(self, seed: int | None = None, **kwargs):
        self.seed = seed

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        from tabpfn import TabPFNRegressor

        self._reg = TabPFNRegressor(random_state=self.seed)
        self._reg.fit(train_x, train_y)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._reg.predict(X_test)


class TabPFNEnsembleModel(SurrogateModel):
    """Bagged ensemble of ``TabPFNRegressor``s over distinct random seeds.

    Each member sees the same data but a different ``random_state`` (which
    permutes TabPFN's internal feature/estimator configuration), so the spread
    across members is a cheap posterior-uncertainty proxy. ``predict`` returns
    the member mean, and with ``return_std=True`` also the member std -- letting
    the ensemble drive variance-/EI-based selection, unlike a single TabPFN.
    """

    def __init__(self, n_members: int = 8, random_state: int = 0, **kwargs):
        self.n_members = kwargs.get("tabpfn_n_members", n_members)
        self.random_state = random_state
        self._regs: list = []

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        from tabpfn import TabPFNRegressor

        n_members = kwargs.get("tabpfn_n_members", self.n_members)
        self._regs = []
        for i in range(n_members):
            reg = TabPFNRegressor(random_state=self.random_state + i)
            reg.fit(train_x, train_y)
            self._regs.append(reg)

    def predict(self, X_test: np.ndarray, return_std: bool = False):
        preds = np.stack([reg.predict(X_test).ravel() for reg in self._regs], axis=0)
        mean = preds.mean(axis=0)
        if return_std:
            return mean, preds.std(axis=0)
        return mean
