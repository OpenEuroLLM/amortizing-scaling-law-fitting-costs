"""``build_fit`` fits a loss form to a grid and returns sane, in-bounds parameters."""
import unittest

import numpy as np

from core.fitting import build_fit
from core.forms import PARAMETRIC_FORMS
from tests._grid import make_grid


class TestBuildFit(unittest.TestCase):
    def test_l0_fit_is_finite_and_in_bounds(self):
        df = make_grid()
        df["C"] = 6.0 * df["N"] * df["D"]  # L0 fits loss vs C off the Pareto frontier
        params, loss = build_fit(df, form_name="L0", fit_quality="debug")

        self.assertEqual(len(params), 3)
        self.assertTrue(np.all(np.isfinite(params)))
        self.assertTrue(np.isfinite(loss))

        interp = PARAMETRIC_FORMS["L0"].interpretable(params)
        self.assertEqual(set(interp), {"L0", "A", "alpha"})
        self.assertGreater(interp["L0"], 0.0)
        self.assertTrue(-1.0 <= interp["alpha"] <= 0.0)  # power-law exponent bounds

    def test_steplaw_lr_fit_runs(self):
        # optimal-HP form: fit the min-loss lr per (N, D) scale; just check it produces finite params.
        df = make_grid()
        optima = df.loc[df.groupby(["N", "D"], sort=False)["loss"].idxmin()]
        params, loss = build_fit(optima, form_name="steplaw_lr", y_name="lr", fit_quality="debug")
        self.assertTrue(np.all(np.isfinite(params)))
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
