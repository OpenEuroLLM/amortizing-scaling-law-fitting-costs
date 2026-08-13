import numpy as np
import pandas as pd

from .base import DataSelection, FEATURES, TARGET


class MaxVarianceSelection(DataSelection):
    """Uncertainty sampling: acquire the candidate where the surrogate is least certain.

    Same loop as ``RandomSelection`` (the driver fits a GP each iteration), but instead of a
    uniform draw it buys the pool point of highest GP posterior standard deviation. The driver
    fits the surrogate on the current observations and passes it in via ``surrogate``; before a
    surrogate is available it falls back to a random draw so the run can bootstrap.
    """
    needs_surrogate = True   # tells the driver to fit + pass a GP before next_point

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def split(
        self,
        df_fit: pd.DataFrame,
        n_point: int,
        rng: np.random.RandomState | None = None
    ) -> tuple[np.ndarray, np.ndarray, float, pd.Index]:
        # method-agnostic random initial batch, identical to RandomSelection.split
        if rng is None:
            rng = np.random.RandomState(self.seed)
        idx = rng.choice(df_fit.index, n_point, replace=False)
        remaining_idx = df_fit.index.difference(idx)
        return (df_fit[FEATURES].iloc[idx].to_numpy(), df_fit[TARGET].iloc[idx].to_numpy(),
                df_fit['C'].iloc[idx].sum(), remaining_idx)

    def next_point(
        self,
        df: pd.DataFrame,
        candidate_set: pd.Index | None = None,
        surrogate=None,
        **kwargs
    ) -> int:
        if len(df) == 0:
            raise ValueError("Dataframe is empty, cannot select next point.")
        if candidate_set is None:
            candidate_set = df.index
        candidate_set = pd.Index(candidate_set)
        if surrogate is None:                       # bootstrap before a GP is available
            return int(self.rng.choice(candidate_set))
        X = np.log(df.loc[candidate_set, FEATURES].to_numpy(float))
        _, std = surrogate.predict(X, return_std=True)
        return int(candidate_set[int(np.argmax(np.asarray(std).ravel()))])
