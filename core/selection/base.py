from abc import ABC, abstractmethod

import pandas as pd

# Selectors operate in the same config-grid space as the surrogate; source the schema
# from the one place (core) rather than redefining it here. Selectors use the `default`
# feature set; the `--feature_set full` extras only feed the GP infill. See core/__init__.py.
from core import DEFAULT_FEATURES as FEATURES, GRID_TARGET as TARGET  # noqa: F401


class DataSelection(ABC):
    @abstractmethod
    def split(self, df_fit: pd.DataFrame, n_points: int) -> tuple[pd.DataFrame, pd.Series]:
        """Return (X, y) with n_points training rows selected from df_fit."""

    @abstractmethod
    def next_point(self, df: pd.DataFrame, candidate_set: pd.Index | None = None, **kwargs) -> int:
        """Return the index of the next point to acquire. candidate_set=None → full df."""
