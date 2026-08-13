"""Acquisition functions over surrogate predictions.

Each acquisition scores candidate points from the surrogate's predictive
(mu, sigma); the driver picks the argmax. We minimize loss, so "improvement"
is over the best (lowest) loss observed so far.
"""
import numpy as np
from scipy.stats import norm


class ExpectedImprovement:
    """EI for minimization: E[max(incumbent - f(x), 0)] under f(x) ~ N(mu, sigma).

    incumbent = best (lowest) observed loss so far.
    """

    def __init__(self, xi: float = 0.0):
        self.xi = xi  # exploration margin; 0 = standard EI

    def score(self, mu: np.ndarray, sigma: np.ndarray, incumbent: float) -> np.ndarray:
        mu = np.asarray(mu, dtype=float).ravel()
        sigma = np.asarray(sigma, dtype=float).ravel()
        sigma = np.clip(sigma, 1e-12, None)          # avoid divide-by-zero
        improvement = incumbent - mu - self.xi        # >0 means expected to beat incumbent
        z = improvement / sigma
        return improvement * norm.cdf(z) + sigma * norm.pdf(z)


class ExpectedImprovementPerCost(ExpectedImprovement):
    """EI per unit cost: EI(x) / log(cost(x)), with cost = 6*N*D (FLOPs)."""

    def score(self, mu: np.ndarray, sigma: np.ndarray, incumbent: float,
              cost: np.ndarray) -> np.ndarray:
        ei = super().score(mu, sigma, incumbent)
        log_cost = np.log(np.asarray(cost, dtype=float).ravel())
        return ei / log_cost
