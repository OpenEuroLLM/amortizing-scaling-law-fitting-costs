"""Registry of parametric L(...) forms usable by the fitting pipeline.

Each ``ParametricForm`` (see ``core.forms.base``) bundles everything ``core.fitting``
and ``main.py`` need to fit and report on a functional form. ``PARAMETRIC_FORMS`` maps
CLI names to the forms.

To add a new form: write a module mirroring ``l0.py`` / ``chinchilla.py``, then import
its ``*_FORM`` here and add it to ``PARAMETRIC_FORMS``.
"""

from .base import ParametricForm
from .l0 import L0_FORM
from .chinchilla import CHINCHILLA3_FORM, CHINCHILLA_INTERACTION_FORM, chinchilla_compute_optimal_ND
from .width_depth import WIDTH_DEPTH_FORM
from .steplaw import STEPLAW_LR_FORM, STEPLAW_BSZ_FORM

PARAMETRIC_FORMS: dict[str, ParametricForm] = {
    "L0": L0_FORM,
    "chinchilla3": CHINCHILLA3_FORM,
    "chinchilla_interaction": CHINCHILLA_INTERACTION_FORM,
    "width_depth": WIDTH_DEPTH_FORM,
    "steplaw_lr": STEPLAW_LR_FORM,
    "steplaw_bsz": STEPLAW_BSZ_FORM,
}

__all__ = [
    "ParametricForm",
    "PARAMETRIC_FORMS",
    "chinchilla_compute_optimal_ND",
    "L0_FORM",
    "CHINCHILLA3_FORM",
    "CHINCHILLA_INTERACTION_FORM",
    "WIDTH_DEPTH_FORM",
    "STEPLAW_LR_FORM",
    "STEPLAW_BSZ_FORM",
]
