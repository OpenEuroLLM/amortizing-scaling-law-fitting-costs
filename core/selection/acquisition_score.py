from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import pandas as pd

from .base import DataSelection
from ..utils.data import DEFAULT_FEATURES


FEATURES = DEFAULT_FEATURES
TARGET = 'loss'


class EISelection(DataSelection):
    """Acquire the pool point with the highest Expected Improvement.

    Minimization: incumbent = best (lowest) observed loss so far. The driver fits a GP
    on the current observations and passes it in via ``surrogate``; before one is
    available (or if it fails) we fall back to a random draw so the run can bootstrap.
    """
    needs_surrogate = True

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        from .acquisition import ExpectedImprovement
        self.acq = ExpectedImprovement()

    def split(self, df_fit, n_points):
        raise NotImplementedError("EISelection is sequential; use next_point.")

    def next_point(
        self,
        df: pd.DataFrame,
        candidate_set: pd.Index | None = None,
        acquired_idx: pd.Index | None = None,
        surrogate=None,
        **kwargs
    ) -> int:
        if candidate_set is None:
            candidate_set = df.index
        candidate_set = pd.Index(candidate_set)
        if surrogate is None or acquired_idx is None or len(acquired_idx) == 0:
            return int(self.rng.choice(candidate_set))     # bootstrap
        incumbent = float(df.loc[acquired_idx, TARGET].min())
        X = np.log(df.loc[candidate_set, FEATURES].to_numpy(float))
        mu, sigma = surrogate.predict(X, return_std=True)
        scores = self.acq.score(mu, sigma, incumbent)
        return int(candidate_set[int(np.argmax(scores))])


class PISelection(EISelection):
    """Acquire the pool point with the highest Probability of Improvement."""

    def __init__(self, seed: int):
        super().__init__(seed)
        from .acquisition import ProbabilityOfImprovement
        self.acq = ProbabilityOfImprovement()


class LCBSelection(EISelection):
    """Acquire the pool point with the lowest LCB (mu - kappa*sigma).

    First acquisition with its own hyperparameter: ``kappa`` (exploration weight).
    LCB needs no incumbent, so next_point is overridden.
    """

    def __init__(self, seed: int, kappa: float = 2.0):
        super().__init__(seed)
        from .acquisition import LowerConfidenceBound
        self.acq = LowerConfidenceBound(kappa=kappa)

    def next_point(
        self,
        df: pd.DataFrame,
        candidate_set: pd.Index | None = None,
        acquired_idx: pd.Index | None = None,
        surrogate=None,
        **kwargs
    ) -> int:
        if candidate_set is None:
            candidate_set = df.index
        candidate_set = pd.Index(candidate_set)
        if surrogate is None:
            return int(self.rng.choice(candidate_set))     # bootstrap
        X = np.log(df.loc[candidate_set, FEATURES].to_numpy(float))
        mu, sigma = surrogate.predict(X, return_std=True)
        scores = self.acq.score(mu, sigma)
        return int(candidate_set[int(np.argmax(scores))])


class EIPerCostSelection(EISelection):
    """Acquire the pool point with the highest EI per unit cost, cost = 6*N*D."""

    def __init__(self, seed: int):
        super().__init__(seed)
        from .acquisition import ExpectedImprovementPerCost
        self.acq = ExpectedImprovementPerCost()

    def next_point(
        self,
        df: pd.DataFrame,
        candidate_set: pd.Index | None = None,
        acquired_idx: pd.Index | None = None,
        surrogate=None,
        **kwargs
    ) -> int:
        if candidate_set is None:
            candidate_set = df.index
        candidate_set = pd.Index(candidate_set)
        if surrogate is None or acquired_idx is None or len(acquired_idx) == 0:
            return int(self.rng.choice(candidate_set))     # bootstrap
        incumbent = float(df.loc[acquired_idx, TARGET].min())
        # X = np.log(df.loc[candidate_set, FEATURES].to_numpy(float))
        X = df.loc[candidate_set, FEATURES].to_numpy(float)
        mu, sigma = surrogate.predict(X, return_std=True)
        cost = df.loc[candidate_set, "C"].to_numpy(float)
        scores = self.acq.score(mu, sigma, incumbent, cost=cost)
        return int(candidate_set[int(np.argmax(scores))])
# end of file