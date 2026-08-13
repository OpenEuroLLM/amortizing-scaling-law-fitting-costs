"""Fit a parametric form to (subsampled or surrogate-infilled) data."""
import numpy as np
import pandas as pd

from scaling.utils import fit_parametric_form, get_pareto_frontier

from .forms import PARAMETRIC_FORMS


def build_fit(
    df: pd.DataFrame,
    form_name: str = "L0",
    y_name: str | None = None,
    fit_quality: str = 'debug',
) -> tuple[np.ndarray, float]:
    """Fit ``PARAMETRIC_FORMS[form_name]`` to ``df``.

    ``y_name`` defaults to the form's ``target`` column. If the form uses the Pareto
    envelope, ``df`` is first reduced to its loss frontier along the first input column.
    """
    form = PARAMETRIC_FORMS[form_name]
    y_name = y_name if y_name is not None else form.target
    fit_df = get_pareto_frontier(df, x_name=form.input_cols[0], y_name=y_name) if form.use_pareto else df
    X_data = fit_df[form.input_cols].to_numpy()
    if X_data.shape[1] == 1:
        X_data = X_data.ravel()
    best_params, best_loss, _ = fit_parametric_form(
        form.func,
        X_data,
        np.log(fit_df[y_name].to_numpy()),
        form.initial_grid(fit_quality),
        form.bounds,
    )
    return best_params, best_loss


def print_fit_results(form, params: np.ndarray) -> None:
    print(form.format_fit(params))
