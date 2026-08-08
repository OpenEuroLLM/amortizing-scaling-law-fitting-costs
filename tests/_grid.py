"""Tiny synthetic N x D x lr x gbsz grid shared by the smoke tests.

The loss is a smooth L0-like surface (irreducible + power law in compute) plus convex penalties
that place a well-defined optimal lr/gbsz at each scale -- enough for both the loss-law and the
optimal-HP (step-law) pipelines to have something to recover, while staying small and fast.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

_NS = [1e6, 3e6, 1e7, 3e7, 1e8]      # 5 model scales (largest is held out by both pipelines)
_DS = [1e8, 3e8, 1e9]
_LRS = [1e-3, 3e-3, 1e-2]
_GS = [64, 256, 1024]


def make_grid() -> pd.DataFrame:
    """The synthetic grid as a DataFrame (N, D, lr, gbsz, loss)."""
    rows = []
    for N in _NS:
        for D in _DS:
            C = 6 * N * D
            lr_opt = 3e-3 * (C / 6e14) ** -0.03   # optimum drifts with scale -> a non-trivial step law
            g_opt = 256 * (D / 1e8) ** 0.1
            for lr in _LRS:
                for g in _GS:
                    loss = (1.0 + 200 * C ** -0.1
                            + 0.05 * (np.log(lr) - np.log(lr_opt)) ** 2
                            + 0.03 * (np.log(g) - np.log(g_opt)) ** 2)
                    rows.append(dict(N=N, D=D, lr=lr, gbsz=g, loss=loss))
    return pd.DataFrame(rows)


def write_grid(path) -> Path:
    """Write the synthetic grid to ``path`` and return it."""
    path = Path(path)
    make_grid().to_csv(path, index=False)
    return path
