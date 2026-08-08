import numpy as np
import pandas as pd

from .base import DataSelection, FEATURES, TARGET


class FarthestPointSelection(DataSelection):
    """Farthest-point (max-min) sampling: acquire the candidate that is farthest, in standardized
    log-feature space, from everything observed so far. A surrogate-free, space-filling design that
    spreads points over the (N, D, lr, gbsz) grid. No GP needed -- pure geometry.

    Before any point is observed it falls back to a random draw so the run can bootstrap.
    """
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
        acquired_idx: pd.Index | None = None,
        **kwargs
    ) -> int:
        if len(df) == 0:
            raise ValueError("Dataframe is empty, cannot select next point.")
        if candidate_set is None:
            candidate_set = df.index
        candidate_set = pd.Index(candidate_set)
        if acquired_idx is None or len(acquired_idx) == 0:
            return int(self.rng.choice(candidate_set))
        obs = np.log(df.loc[acquired_idx, FEATURES].to_numpy(float))
        cand = np.log(df.loc[candidate_set, FEATURES].to_numpy(float))
        mu, sd = obs.mean(0), obs.std(0) + 1e-9         # standardize by the observed set
        obs, cand = (obs - mu) / sd, (cand - mu) / sd
        dmin = np.sqrt(((cand[:, None, :] - obs[None, :, :]) ** 2).sum(-1)).min(1)  # nearest observed
        return int(candidate_set[int(np.argmax(dmin))])                            # farthest such point
