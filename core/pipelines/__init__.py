"""Task pipelines dispatched by ``main.py``.

Each pipeline module exposes ``add_args(parser)`` (register its CLI flags on a subparser) and
``run(args)`` (execute). ``PIPELINES`` maps the ``main.py`` subcommand name to the module.

- ``loss_law`` -- recover a loss scaling law (L vs compute / N,D) via the surrogate acquisition loop.
- ``hp_law``   -- recover an optimal-HP scaling law (optimal lr/gbsz vs scale; step law).
"""
from . import loss_law, hp_law

PIPELINES = {
    "loss_law": loss_law,
    "hp_law": hp_law,
}

__all__ = ["PIPELINES", "loss_law", "hp_law"]
