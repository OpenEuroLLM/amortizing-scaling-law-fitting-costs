import numpy as np
import pandas as pd

from .base import FEATURES


def _sh_schedule(n0, B, eta, n_rounds=None):
    """Deterministic, budget-conserving Successive Halving schedule.

    Returns the per-round active-set sizes, per-model compute increments, and
    the cumulative compute a surviving model has reached after each round. Each
    round spends ``B / R`` in total (so the run spends ``B``); as the field
    halves, survivors receive proportionally more compute.
    """
    R = n_rounds or max(1, int(np.ceil(np.log(n0) / np.log(eta))))
    n_active, increment, n = [], [], n0
    for _ in range(R):
        n_active.append(n)
        increment.append(B / (n * R))
        n = max(1, int(np.floor(n / eta)))
    return R, np.array(n_active), np.array(increment), np.cumsum(increment)


class SuccessiveHalving:
    """Compute-budget *allocation* for scaling-law estimation (arXiv:2605.17234).

    Unlike the selectors above -- which subsample rows of a fixed grid via
    ``split`` -- Successive Halving decides *how much compute to spend on each*
    of a set of model sizes, reading their learning curves from an
    ``oracle(N, C) -> loss``. It is the repo's third literature baseline, but it
    speaks a different interface (allocation, not grid row-selection), so it
    deliberately does not subclass ``DataSelection``.

    Strategies:

    - ``uniform`` -- split ``B`` equally across all models, no pruning.
    - ``run`` -- SH: rounds of train-a-bit / keep-the-best-half, ranking
      survivors by their *observed* (possibly noisy) loss.
    - ``run(surrogate=...)`` -- SH + surrogate: rank by the surrogate's
      *predicted final* loss, so a model that merely looks bad early is not
      pruned. ``surrogate(obs, model_Ns, C_target) -> loss at C_target``, where
      ``obs`` is the list of ``(N, C, loss)`` revealed so far.

    Each strategy returns a dict: ``curves`` (per-model ``{N, C_end, Cs, Ls}``),
    ``spent`` compute, ``l_star`` (lowest loss reached), and ``active`` (the
    models kept for the final round). Losses in ``curves``/``l_star`` are the
    true (clean) oracle values; ``noise`` corrupts only what selection observes.
    """

    def __init__(self, seed: int = 0, eta: float = 2.0,
                 n_points: int = 20, C_floor_frac: float = 0.05):
        self.seed = seed
        self.eta = eta
        self.n_points = n_points
        self.C_floor_frac = C_floor_frac

    def _curve(self, N, C_end, C_floor, oracle):
        """A model's learning curve, sampled from a small floor up to ``C_end``."""
        Cs = np.geomspace(C_floor, C_end, self.n_points)
        return dict(N=float(N), C_end=float(C_end), Cs=Cs, Ls=np.asarray(oracle(N, Cs), float))

    @staticmethod
    def _finish(curves, spent, method, active=None):
        finals = np.array([c["Ls"][-1] for c in curves])
        return dict(method=method, curves=curves, spent=float(spent),
                    l_star=float(finals.min()), active=active)

    def uniform(self, models, B, oracle):
        C_end = B / len(models)
        curves = [self._curve(N, C_end, C_end * self.C_floor_frac, oracle) for N in models]
        return self._finish(curves, len(models) * C_end, "uniform", active=None)

    def run(self, models, B, oracle, surrogate=None, noise=0.0):
        models = np.asarray(models, float)
        n0 = len(models)
        R, n_active, increment, cumulative = _sh_schedule(n0, B, self.eta)
        C_floor = cumulative[0] * self.C_floor_frac
        rng = np.random.RandomState(self.seed)

        active = list(range(n0))
        C_reached = {i: cumulative[0] for i in active}
        obs = []                                 # noisy (N, C, loss) revealed so far -- what selection sees
        spent = 0.0
        for r in range(R):
            C_cum = cumulative[r]
            obs_now = {}                         # observed (noisy) loss at this round's checkpoint
            for i in active:
                C_reached[i] = C_cum
                spent += increment[r]
                seg = np.geomspace(cumulative[r - 1] if r else C_floor, C_cum, max(2, self.n_points // R))
                L = np.asarray(oracle(models[i], seg), float)
                if noise:
                    L = L + rng.normal(0.0, noise, size=L.shape)
                obs.extend(zip(np.full_like(seg, models[i]), seg, L))
                obs_now[i] = L[-1]
            if r == R - 1 or len(active) <= 1:
                break
            # keep top-k by observed loss (SH) or the surrogate's predicted final loss.
            if surrogate is None:
                score = np.array([obs_now[i] for i in active])
            else:
                score = surrogate(obs, models[active], cumulative[-1])
            active = [active[j] for j in np.argsort(score)[: int(n_active[r + 1])]]

        curves = [self._curve(models[i], C_reached[i], C_floor, oracle) for i in range(n0)]
        return self._finish(curves, spent, "SH" if surrogate is None else "SH-surrogate", active=active)


class SuccessiveHalvingGP:
    """Successive Halving as a GP-guided acquisition step (adaptation of arXiv:2605.17234).

    A sequential form of the paper's SH-with-surrogate. Instead of batch rungs, each
    ``next_point`` advances the *most promising* learning curve by one rung and reads a
    real grid row -- so it plugs into the same acquisition loop as the other baselines
    and pays for hyperparameters like they do (no free oracle HP).

    A "model" is a size ``N``; its rungs are increasing ``D`` (more tokens = more
    compute). Given the loop's fitted GP surrogate, each step:

    1. considers only *unopened* ``(N, D)`` cells (rungs not yet observed);
    2. scores each size by the GP's best achievable predicted loss over its remaining
       rungs -- its *potential* -- and keeps the best size (the "halving");
    3. advances that size to its next-larger ``D``, letting the **GP pick** the
       ``(lr, gbsz)`` config to actually run there.

    Falls back to a random draw until there is a surrogate and some observations.

    ``explore`` in [0, 1] adds a coverage term: with that probability a step opens
    the unobserved rung whose compute is *farthest* (in log-C) from everything seen,
    widening the compute range. Pure exploitation (``explore=0``) concentrates on the
    promising size and pins the frontier but leaves the range -- and the fitted
    exponent -- poorly covered; a little exploration trades some of that for a
    better-determined scaling law.
    """

    def __init__(self, seed: int = 0, explore: float = 0.0):
        self.seed = seed
        self.explore = explore
        self.rng = np.random.RandomState(seed)

    def next_point(self, df, candidate_set=None, acquired_idx=None, surrogate=None, **kwargs) -> int:
        candidate_set = df.index if candidate_set is None else pd.Index(candidate_set)
        cand = df.loc[candidate_set]
        if surrogate is None or acquired_idx is None or len(acquired_idx) == 0:
            return int(self.rng.choice(candidate_set))

        # prefer opening a new (N, D) rung over re-sampling an already-observed cell
        observed_cells = set(map(tuple, df.loc[acquired_idx, ["N", "D"]].to_numpy()))
        fresh = [(n, d) not in observed_cells for n, d in cand[["N", "D"]].to_numpy()]
        pool = cand[fresh] if any(fresh) else cand

        pred = np.asarray(surrogate.predict(np.log(pool[FEATURES].to_numpy(float)))).ravel()
        pool = pool.assign(_pred=pred)

        if self.explore and self.rng.rand() < self.explore:
            # coverage move: open the rung whose compute is farthest from what we've seen
            seen = np.log(df.loc[acquired_idx, "C"].to_numpy(float))
            gap = np.array([np.min(np.abs(np.log(c) - seen)) for c in pool["C"].to_numpy(float)])
            tgt = pool.iloc[int(np.argmax(gap))]
            cellrows = pool[(pool.N == tgt.N) & (pool.D == tgt.D)]
            return int(cellrows["_pred"].idxmin())       # GP picks the config in that far cell

        promise = pool.groupby("N")["_pred"].min()       # best loss each size could still reach
        best_size = promise.idxmin()                     # keep the most promising size
        sub = pool[pool.N == best_size]
        sub = sub[sub.D == sub.D.min()]                  # advance its curve by one rung
        return int(sub["_pred"].idxmin())                # GP picks the config at that rung
