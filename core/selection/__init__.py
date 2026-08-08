"""Training-point selection strategies.

``SELECTION`` maps CLI names to the row-subsampling selectors that plug into the
acquisition loop (``split`` + ``next_point``). ``SuccessiveHalving`` /
``SuccessiveHalvingGP`` are literature baselines with their own interfaces and are
exported directly rather than via the registry.
"""

from .base import DataSelection, FEATURES, TARGET
from .random import RandomSelection
from .max_variance import MaxVarianceSelection
from .farthest_point import FarthestPointSelection
from .local_search import LocalSearch
from .successive_halving import SuccessiveHalving, SuccessiveHalvingGP, _sh_schedule

SELECTION = {
    "random": RandomSelection,
    "local_search": LocalSearch,
    "max_variance": MaxVarianceSelection,
    "farthest_point": FarthestPointSelection,
}

__all__ = [
    "DataSelection",
    "FEATURES",
    "TARGET",
    "RandomSelection",
    "MaxVarianceSelection",
    "FarthestPointSelection",
    "LocalSearch",
    "SuccessiveHalving",
    "SuccessiveHalvingGP",
    "SELECTION",
]
