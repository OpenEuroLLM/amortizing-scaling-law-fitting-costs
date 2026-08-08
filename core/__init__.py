"""Surrogate-based scaling laws: core library.

Subpackages:
- ``core.selection``  -- training-point selection strategies (+ SELECTION registry)
- ``core.surrogates`` -- surrogate models (+ MODELS registry, build_surrogate)
- ``core.forms``      -- parametric L(...) forms (+ PARAMETRIC_FORMS registry)
- ``core.fitting``    -- fit a parametric form to (subsampled) data
- ``core.utils``      -- data preprocessing helpers and plotting styles

The single entry point is ``main.py`` at the repo root.
"""

# The features the GP conditions on, and the quantity it learns. The GP always models
# `loss`; which *features* it sees is chosen by `--feature_set`, to compare whether the
# architectural HPs help loss-infill and law recovery. `core.utils.data.resolve_grid_features`
# maps the mode to actual columns:
#
#   default -- the aggregate config axes: N, D, lr, gbsz.
#   arch    -- default + the architectural HPs (ARCH_FEATURES).
#   data    -- default + the data-axis HPs (DATA_FEATURES).
#   full    -- default + both. `arch`/`data` isolate each group's contribution.
#
# All groups are auto-narrowed by `resolve_grid_features` to columns that are present and
# carry actual signal (not entirely null, not constant): datasets with all-NaN lr/gbsz
# (HP-less sources), non-`--extended` CSVs (no arch/data columns), or a study that holds
# a knob fixed (constant column) simply drop those -- `full` collapses toward `default`.
#
# The ARCH_/DATA_FEATURES names are preprocess.py's *canonical* schema (it renames each
# source's raw columns to these), so these lists are dataset-agnostic: a dataset picks up
# whichever it carries. Add a name only if a new source introduces a differently-named knob
# (e.g. MoE's expert_width).
#
# These are the GP's INPUT features, NOT the parametric fit's input/target -- that varies
# per form and lives on each `ParametricForm` as `input_cols` / `target` (e.g. L0 fits
# `loss` vs `C`). `build_fit` reads those off the form; don't conflate the two.
DEFAULT_FEATURES = ["N", "D", "lr", "gbsz"]
ARCH_FEATURES = ["width", "depth", "ffn", "num_heads"]   # architectural HPs
DATA_FEATURES = ["unique_tokens", "seq_len"]             # data-axis HPs (repetition / context length)
GRID_TARGET = "loss"

# Safety cap on the number of acquisition iterations when no budget is given.
MAX_ITERS = 1000
