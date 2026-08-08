"""Data preprocessing helpers for the acquisition loop."""
from pathlib import Path

import numpy as np
import pandas as pd

from core import DEFAULT_FEATURES, ARCH_FEATURES, DATA_FEATURES


def clean_grid(src, dst=None):
    """Driver-ready copy of a raw grid CSV: rename ``bs``->``gbsz`` and cast ``N,D`` to float.

    The float cast avoids int64 ``C = 6ND`` overflowing to negatives at large scales. Rows missing
    any of ``N, D, lr, gbsz, loss`` are dropped; any extra columns (e.g. ``width``/``depth`` for the
    ``arch`` feature set) are preserved. Writes ``<stem>_clean.csv`` next to ``src`` unless ``dst``
    is given, and returns the path.
    """
    src = Path(src)
    df = pd.read_csv(src).rename(columns={"bs": "gbsz"})
    df = df.dropna(subset=["N", "D", "lr", "gbsz", "loss"])
    df[["N", "D"]] = df[["N", "D"]].astype(float)
    dst = Path(dst) if dst else src.with_name(src.stem + "_clean.csv")
    df.to_csv(dst, index=False)
    return dst


FEATURE_SETS = ("default", "arch", "data", "full")


def resolve_grid_features(df: pd.DataFrame, feature_set: str = "default") -> list[str]:
    """Columns the GP conditions on for this dataset, by mode.

    ``default`` -> DEFAULT_FEATURES (N, D, lr, gbsz);
    ``arch``    -> default + ARCH_FEATURES (architectural HPs);
    ``data``    -> default + DATA_FEATURES (data-axis HPs);
    ``full``    -> default + both. ``arch``/``data`` isolate each group's contribution.

    Every candidate is auto-narrowed to columns that carry actual signal -- present, not
    entirely null, and not constant. So HP-less datasets (all-NaN lr/gbsz), non-``--extended``
    CSVs (no arch/data columns), and studies that hold a knob fixed (constant column) all drop
    those; the extra modes collapse toward ``default`` when nothing extra survives.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}' (expected one of {FEATURE_SETS}).")
    wanted = list(DEFAULT_FEATURES)
    if feature_set in ("arch", "full"):
        wanted += ARCH_FEATURES
    if feature_set in ("data", "full"):
        wanted += DATA_FEATURES
    # keep only columns with >= 2 distinct non-null values (drops absent, all-null, constant)
    return [c for c in wanted if c in df.columns and df[c].nunique(dropna=True) > 1]


def next_batch_compute(
    df: pd.DataFrame,
    acquired_idx: pd.Index | None = None,
    C_max: float | None = None,
    C_window: tuple[float, float] | None = None,
    C_envelope_growth: float = 2.0
) -> pd.DataFrame:
    """Subsets the data within a compute bracket."""
    pool = df.index.difference(acquired_idx) if acquired_idx is not None else df.index
    if C_max is not None and C_window is not None:
        while True:
            _idxs = (df.loc[pool, "C"] >= C_window[0] * C_max) & (df.loc[pool, "C"] <= C_window[1] * C_max)
            if _idxs.any():
                pool = pool[_idxs]
                break
            elif C_max > df.loc[pool, "C"].max():
                # if the max compute is already above the envelope, just return the pool
                break
            else:
                # push the envelope to not return an empty pool
                C_max *= C_envelope_growth
    return df.loc[pool]


def data_preprocess(
    df: pd.DataFrame,
    held_out_type: str = 'N',
    held_out_C_lower: float = 0.5,
    C_extrapolation_len: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["C"] = 6 * df["N"] * df["D"]

    # Keep the max compute as the test point
    if held_out_type == 'N':
        test_df_known = df.loc[df.N == df.N.max()]
        _upper_lim, _col = df.N.max(), "N"
    elif held_out_type == 'C':
        _pseudo_C_max = df.C.max() * held_out_C_lower
        test_df_known = df.loc[df.C >= _pseudo_C_max]
        _upper_lim, _col = _pseudo_C_max, "C"
    else:
        raise ValueError(f"Unsupported held_out_type: {held_out_type}")

    # Create an extrapolation limit as unknown test set
    test_df_unknown = pd.DataFrame.from_dict({
        "C": [test_df_known.C.max() * 10 ** i for i in range(1, C_extrapolation_len + 1)],
        "N": [None] * C_extrapolation_len,
        "D": [None] * C_extrapolation_len,
        "lr": [None] * C_extrapolation_len,
        "gbsz": [None] * C_extrapolation_len,
        "loss": [None] * C_extrapolation_len,
    })

    train_df = df.loc[df[_col] < _upper_lim]

    return train_df, test_df_known, test_df_unknown


def iterative_data_splits(
    df: pd.DataFrame,
    splitter,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, float, pd.Index]:
    X, y, total_compute, remaining_idx = splitter.split(df, n_points)
    return X, y, total_compute, remaining_idx
