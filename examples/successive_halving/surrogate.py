"""Surrogate for SH+surrogate: predict where a learning curve is *heading*.

The paper uses a multitask GP (LMC) / Deep Ensembles to extrapolate each
model's learning curve and pick the model with the most promising *future*
loss. We use a single joint GP over ``(log N, log C) -> log loss`` fit on all
revealed points: it borrows strength across curves (cross-curve correlation,
the point of the LMC) without committing to the generating functional form.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models import GPModel


def gp_surrogate(obs_points, model_Ns, C_target):
    """Predicted loss of each model in ``model_Ns`` if trained to compute ``C_target``.

    ``obs_points`` is an iterable of ``(N, C, loss)`` revealed so far.
    """
    obs = np.asarray(list(obs_points), dtype=float)      # (n, 3): N, C, loss
    X = np.log(obs[:, :2])
    y = np.log(obs[:, 2:3])
    gp = GPModel()
    gp.fit(X, y)
    Xq = np.column_stack([np.log(np.asarray(model_Ns, float)),
                          np.full(len(model_Ns), np.log(C_target))])
    return np.exp(np.asarray(gp.predict(Xq)).ravel())
