"""Fit a scaling law to a strategy's learning curves and score it.

Two things matter (paper Eqs. 1-2):

- ``l_star`` -- the lowest loss any trained model reached (the proxy objective).
- the loss-compute *frontier* the curves trace, which we fit ``L(C)=E+A*C**b``
  to and compare against the gold-standard frontier of the true ``L(N, D)``.
"""
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from synth_lc import true_frontier


def observed_frontier(curves):
    """Lower envelope of all curves: the running-min loss over increasing compute."""
    Cs = np.concatenate([c["Cs"] for c in curves])
    Ls = np.concatenate([c["Ls"] for c in curves])
    order = np.argsort(Cs)
    Cs, Ls = Cs[order], Ls[order]
    keep = np.minimum.accumulate(Ls[::-1])[::-1]         # loss achievable by *some* model at >= this C
    # a compute is on the frontier if its loss equals the best loss from there on
    mask = Ls <= keep + 1e-12
    C_f, L_f = Cs[mask], Ls[mask]
    # collapse to a strictly-decreasing envelope
    strict = np.concatenate([[True], np.diff(np.minimum.accumulate(L_f)) < 0])
    return C_f[strict], np.minimum.accumulate(L_f)[strict]


def _law(C, E, logA, b):
    return E + np.exp(logA) * C ** b


def fit_law(C, L):
    """Fit ``L(C) = E + A*C**b`` on a frontier; returns ``(E, A, b)``.

    Multi-start over a small grid of (E, b) seeds and keep the best SSE -- the fit is
    ill-conditioned on few, short-range frontier points, so a single start is fragile.
    """
    C = np.asarray(C, float); L = np.asarray(L, float)
    best, best_sse = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)   # covariance of a good fit can be singular
        for E0 in (0.5, 1.0, 1.5, 2.0):
            for b0 in (-0.05, -0.1, -0.2, -0.5):
                try:
                    popt, _ = curve_fit(_law, C, L, p0=[E0, np.log(1e3), b0], maxfev=40000,
                                        bounds=([0.0, -20.0, -1.0], [max(L) + 1e-9, 40.0, 0.0]))
                except Exception:
                    continue
                sse = float(np.sum((_law(C, *popt) - L) ** 2))
                if sse < best_sse:
                    best, best_sse = popt, sse
    if best is None:
        raise RuntimeError("fit_law: all starts failed")
    return best[0], float(np.exp(best[1])), best[2]


def evaluate(result, C_ref=None):
    """Attach frontier, fitted law, and gold-standard fit error to a strategy result."""
    C_f, L_f = observed_frontier(result["curves"])
    E, A, b = fit_law(C_f, L_f)
    if C_ref is None:
        C_ref = np.geomspace(C_f.min(), C_f.max(), 100)
    L_true = true_frontier(C_ref)
    L_hat = _law(C_ref, E, np.log(A), b)
    frontier_mse = float(np.mean((np.log(L_hat) - np.log(L_true)) ** 2))   # log-space, like the fits
    return {**result, "frontier": (C_f, L_f), "law": (E, A, b),
            "frontier_mse": frontier_mse}
