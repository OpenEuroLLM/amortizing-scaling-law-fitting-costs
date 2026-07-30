#!/usr/bin/env python3
"""Write a dense synthetic grid CSV for the acquisition comparison.

Unlike steplaw (5 sizes), this spans many model sizes over a wide N range, so the
L(N,D) exponents alpha/beta are actually identifiable. Loss = Chinchilla L(N,D)
plus a small hyperparameter penalty (3 configs per cell, middle is best) plus
noise -- so `SuccessiveHalvingGP` still has a real (lr, gbsz) choice to make.

    uv run python examples/successive_halving/make_synth_grid.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_lc import loss_ND

OUT = REPO / "results" / "baselines" / "successive_halving" / "data" / "synth_grid.csv"

N_SIZES = 2.0 ** np.linspace(16, 40, 16)          # wide range -> alpha identifiable
D_VALUES = np.geomspace(1e8, 1e11, 8)             # 8 durations per model
CONFIGS = [(3e-4, 128), (1e-3, 256), (3e-3, 512)]  # (lr, gbsz); middle is best
NOISE = 0.02


def main() -> int:
    rng = np.random.RandomState(0)
    rows = []
    for N in N_SIZES:
        for D in D_VALUES:
            base = float(loss_ND(N, D))
            for i, (lr, gbsz) in enumerate(CONFIGS):
                penalty = 0.05 * abs(i - 1)        # hyperparameter cost, 0 at the best config
                loss = base + penalty + rng.normal(0.0, NOISE)
                rows.append(dict(N=N, D=D, lr=lr, gbsz=gbsz, loss=loss))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} rows: {len(N_SIZES)} N x {len(D_VALUES)} D x {len(CONFIGS)} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
