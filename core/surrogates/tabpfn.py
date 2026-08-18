import numpy as np

from .base import SurrogateModel


# ±1 standard-deviation quantiles of a standard normal: for a Gaussian the half
# gap (q_hi - q_lo)/2 equals sigma exactly, and stays a robust scale estimate for
# the (possibly skewed) TabPFN bar distribution.
_SIGMA_QUANTILES = (0.15865525393145707, 0.8413447460685429)


class TabPFNBOModel(SurrogateModel):
    """TabPFN as a Bayesian-optimization surrogate (PriorLabs cookbook).

    Uses TabPFN's *full predictive distribution* -- not just a point prediction --
    so ``predict(..., return_std=True)`` returns a genuine posterior (mean, std),
    the ingredients Expected Improvement needs. ``FullSupportBarDistribution``
    exposes no public variance, so std is read off the +/-1 sigma quantiles
    (``output_type="quantiles"``), which is version-stable across TabPFN releases.

    Pair with ``--selection ei`` / ``ei_per_cost`` to reproduce the cookbook's
    TabPFN + EI loop over this repo's discrete candidate pool. ``n_estimators``
    (>1) averages several internal permutations for a smoother posterior.
    """

    def __init__(self, n_estimators: int = 8, random_state: int = 0,
                 seed: int | None = None, **kwargs):
        self.n_estimators = kwargs.get("tabpfn_n_estimators", n_estimators)
        # the pipeline threads --seed through as `seed`; honor it for reproducibility
        self.random_state = seed if seed is not None else random_state

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        from tabpfn import TabPFNRegressor

        n_estimators = kwargs.get("tabpfn_n_estimators", self.n_estimators)
        self._reg = TabPFNRegressor(
            n_estimators=n_estimators, random_state=self.random_state
        )
        self._reg.fit(train_x, train_y.ravel())

    def predict(self, X_test: np.ndarray, return_std: bool = False):
        mean = np.asarray(self._reg.predict(X_test)).ravel()
        if not return_std:
            return mean
        q_lo, q_hi = self._reg.predict(
            X_test, output_type="quantiles", quantiles=list(_SIGMA_QUANTILES)
        )
        std = (np.asarray(q_hi).ravel() - np.asarray(q_lo).ravel()) / 2.0
        return mean, np.clip(std, 1e-9, None)

    def expected_improvement(
        self, X_test: np.ndarray, incumbent: float, n_quantiles: int = 64
    ) -> np.ndarray:
        """Exact EI over TabPFN's full (non-Gaussian) bar distribution.

        Minimization: ``EI(x) = E[max(incumbent - Y, 0)]`` with ``Y`` the
        posterior at ``x``. We integrate against the true distribution by
        drawing ``n_quantiles`` evenly-spaced (midpoint) quantile samples -- an
        unbiased estimate exact in the ``n_quantiles -> inf`` limit -- rather
        than collapsing to a Gaussian as ``predict(return_std=True)`` does. Uses
        only the documented quantiles API, so it is stable across TabPFN
        releases. Consumed by the ``tabpfn_ei`` selector.
        """
        taus = (np.arange(1, n_quantiles + 1) - 0.5) / n_quantiles
        qs = self._reg.predict(
            X_test, output_type="quantiles", quantiles=list(taus)
        )
        Y = np.stack([np.asarray(q, dtype=float).ravel() for q in qs], axis=0)
        return np.clip(incumbent - Y, 0.0, None).mean(axis=0)


class TabPFNEnsembleModel(SurrogateModel):
    """Bagged ensemble of ``TabPFNRegressor``s over distinct random seeds.

    Each member sees the same data but a different ``random_state`` (which
    permutes TabPFN's internal feature/estimator configuration), so the spread
    across members is a cheap posterior-uncertainty proxy. ``predict`` returns
    the member mean, and with ``return_std=True`` also the member std -- letting
    the ensemble drive variance-/EI-based selection, unlike a single TabPFN.
    """

    def __init__(self, n_members: int = 8, random_state: int = 0,
                 seed: int | None = None, **kwargs):
        self.n_members = kwargs.get("tabpfn_n_members", n_members)
        # the pipeline threads --seed through as `seed`; honor it for reproducibility
        self.random_state = seed if seed is not None else random_state
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
