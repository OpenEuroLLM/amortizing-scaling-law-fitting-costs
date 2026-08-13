import numpy as np
import pandas as pd

from .base import DataSelection, FEATURES, TARGET


class RandomSelection(DataSelection):
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def split(
        self,
        df_fit: pd.DataFrame,
        n_point: int,
        rng: np.random.RandomState | None = None
    ) -> tuple[np.ndarray, np.ndarray, float, pd.Index]:
        if rng is None:
            rng = np.random.RandomState(self.seed)
        idx = rng.choice(df_fit.index, n_point, replace=False)
        remaining_idx = df_fit.index.difference(idx)
        return df_fit[FEATURES].iloc[idx].to_numpy(), df_fit[TARGET].iloc[idx].to_numpy(), df_fit['C'].iloc[idx].sum(), remaining_idx

    def next_point(
        self,
        df: pd.DataFrame,
        candidate_set: pd.Index | None = None,
        **kwargs
    ) -> int:
        if len(df) == 0:
            raise ValueError("Dataframe is empty, cannot select next point.")
        if candidate_set is None:
            candidate_set = df.index
        return self.rng.choice(candidate_set, 1)[0]
