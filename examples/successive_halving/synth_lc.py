"""Synthetic learning-curve data for the Successive Halving baseline.

Loss follows the Chinchilla / Hoffmann ``L(N, D)`` form

    L(N, D) = E + A * N**-alpha + B * D**-beta,        C = 6 * N * D,

so a size-``N`` model traced against compute ``C`` has the learning curve

    L_N(C) = E + A * N**-alpha + B * (C / (6 * N))**-beta.

Bigger models reach a lower loss eventually but need more compute to get there
-- exactly the trade-off Successive Halving has to navigate (small models look
good early, then plateau; big models start poorly, then overtake).
"""
import numpy as np

# Hoffmann et al. (2022), the "Chinchilla" coefficients.
HOFFMANN = dict(A=406.4, B=410.7, E=1.69, alpha=0.34, beta=0.28)


def loss_ND(N, D, coeff=HOFFMANN):
    N, D = np.asarray(N, float), np.asarray(D, float)
    return coeff["E"] + coeff["A"] * N ** -coeff["alpha"] + coeff["B"] * D ** -coeff["beta"]


def loss_at_compute(N, C, coeff=HOFFMANN):
    """Loss of a size-``N`` model trained to compute ``C`` (via ``D = C / 6N``)."""
    C = np.asarray(C, float)
    D = C / (6.0 * np.asarray(N, float))
    return loss_ND(N, D, coeff)


def sample_models(n_models, seed, log2N_range=(10, 34)):
    """Sample ``n_models`` model sizes, log-uniform in parameter count.

    The paper draws sizes from ``2**[2..42]``; we default to a narrower, more
    LLM-realistic band but keep the same log-uniform-in-``N`` spirit.
    """
    rng = np.random.RandomState(seed)
    return np.sort(2.0 ** rng.uniform(*log2N_range, size=n_models))


def true_frontier(C, coeff=HOFFMANN, n_grid=400, log2N_range=(6, 42)):
    """The gold-standard compute-optimal loss at each compute in ``C``.

    For every compute value we minimise ``L_N(C)`` over a dense grid of model
    sizes -- the frontier a fully-trained sweep would trace, i.e. the reference
    an efficient allocation is trying to recover cheaply.
    """
    C = np.asarray(C, float)
    Ngrid = 2.0 ** np.linspace(*log2N_range, n_grid)
    # L[i, j] = loss of model Ngrid[j] at compute C[i]; take the per-row min.
    L = loss_at_compute(Ngrid[None, :], C[:, None], coeff)
    return L.min(axis=1)
