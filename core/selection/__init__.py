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

from .acquisition_score import EISelection, EIPerCostSelection, PISelection, LCBSelection
from .tabpfn_ei import TabPFNEISelection

SELECTION = {
    "random": RandomSelection,
    "local_search": LocalSearch,
    "max_variance": MaxVarianceSelection,
    "farthest_point": FarthestPointSelection,
    "ei": EISelection,
    "ei_per_cost": EIPerCostSelection,
    "pi": PISelection,
    "lcb": LCBSelection,
    "tabpfn_ei": TabPFNEISelection,
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
    "EISelection",
    "EIPerCostSelection",
    "PISelection",
    "LCBSelection",
    "TabPFNEISelection",
]
