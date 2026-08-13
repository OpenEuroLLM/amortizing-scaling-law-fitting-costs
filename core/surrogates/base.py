from abc import ABC, abstractmethod

import numpy as np


class SurrogateModel(ABC):
    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """train_x and train_y are already log-transformed."""

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """X_test is already log-transformed. Returns 1-D predictions."""


class NullSurrogate(SurrogateModel):
    """The model-free choice (``--model_type none``): no surrogate at all.

    ``predict`` returns NaN for every query, so the acquisition loop drops the
    unobserved grid and fits ``L(C)`` on the observed compute-optimal points only.
    Fitting on the (observed + all-NaN) grid is therefore identical to fitting on
    the observations alone -- model-free recovery, with no branch in the loop.
    """

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, *args, **kwargs) -> None:
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.full(len(X_test), np.nan)
