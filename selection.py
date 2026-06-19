from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

FEATURES = ['N', 'D', 'lr', 'gbsz']
TARGET = 'loss'


class DataSelection(ABC):
    @abstractmethod
    def split(self, df_fit: pd.DataFrame, n_points: int) -> tuple[pd.DataFrame, pd.Series]:
        """Return (X, y) with n_points training rows selected from df_fit."""


class RandomSelection(DataSelection):
    def __init__(self, seed: int):
        self.seed = seed

    def split(self, df_fit, n_points):
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(df_fit.index, n_points, replace=False)
        return df_fit[FEATURES].iloc[idx], df_fit[TARGET].iloc[idx]


SELECTION = {
    "random": RandomSelection,
}
