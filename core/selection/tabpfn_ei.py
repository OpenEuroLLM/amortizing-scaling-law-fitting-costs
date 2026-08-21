import numpy as np
import pandas as pd

from .base import DataSelection
from .ei import FEATURES, TARGET
from .acquisition import ExpectedImprovement


class TabPFNEISelection(DataSelection):
    """Acquire the pool point with the highest Expected Improvement, computed
    *exactly* over TabPFN's full bar distribution (PriorLabs BO cookbook).

    Where ``ei`` uses the Gaussian closed-form EI on a collapsed ``(mean, std)``,
    this asks the surrogate for EI integrated over its true, non-Gaussian
    predictive distribution via ``expected_improvement`` -- the faithful
    TabPFN + EI Bayesian-optimization loop, restricted to this repo's discrete
    candidate pool. Best paired with ``--model_type tabpfn_bo``; if the surrogate
    does not expose ``expected_improvement`` (e.g. a GP) it falls back to
    Gaussian EI on ``(mu, sigma)`` so the selector still runs.
    """
    needs_surrogate = True

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._gauss = ExpectedImprovement()

    def split(self, df_fit, n_points):
        raise NotImplementedError("TabPFNEISelection is sequential; use next_point.")

    def next_point(
        self,
        df: pd.DataFrame,
        candidate_set: pd.Index | None = None,
        acquired_idx: pd.Index | None = None,
        surrogate=None,
        **kwargs,
    ) -> int:
        if candidate_set is None:
            candidate_set = df.index
        candidate_set = pd.Index(candidate_set)
        if surrogate is None or acquired_idx is None or len(acquired_idx) == 0:
            return int(self.rng.choice(candidate_set))     # bootstrap
        incumbent = float(df.loc[acquired_idx, TARGET].min())
        X = np.log(df.loc[candidate_set, FEATURES].to_numpy(float))
        if hasattr(surrogate, "expected_improvement"):
            scores = surrogate.expected_improvement(X, incumbent)   # exact bar-dist EI
        else:
            mu, sigma = surrogate.predict(X, return_std=True)        # Gaussian fallback
            scores = self._gauss.score(mu, sigma, incumbent)
        return int(candidate_set[int(np.argmax(scores))])
