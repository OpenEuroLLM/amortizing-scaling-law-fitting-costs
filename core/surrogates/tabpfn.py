import numpy as np

from .base import SurrogateModel


class TabPFNModel(SurrogateModel):
    def fit(self, train_x: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        from tabpfn import TabPFNRegressor

        self._reg = TabPFNRegressor()
        self._reg.fit(train_x, train_y)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._reg.predict(X_test)
