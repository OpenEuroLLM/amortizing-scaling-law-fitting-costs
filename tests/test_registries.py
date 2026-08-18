"""The three registries populate and expose the contract the pipelines rely on.

These are wiring checks: a big reorg's main risk is an import or registration breaking, not the
math. They stay light -- no heavy backend (torch/tabpfn/autogluon) is instantiated.
"""
import unittest

import numpy as np

from core.forms import PARAMETRIC_FORMS, ParametricForm
from core.surrogates import MODELS, NullSurrogate
from core.selection import SELECTION


class TestFormsRegistry(unittest.TestCase):
    def test_expected_forms_present(self):
        self.assertEqual(
            set(PARAMETRIC_FORMS),
            {"L0", "chinchilla3", "chinchilla_interaction", "width_depth", "steplaw_lr", "steplaw_bsz"},
        )

    def test_loss_vs_hp_partition(self):
        # loss_law handles target=="loss"; hp_law handles the rest. The two tasks split on this.
        loss = {n for n, f in PARAMETRIC_FORMS.items() if f.target == "loss"}
        hp = {n for n, f in PARAMETRIC_FORMS.items() if f.target != "loss"}
        self.assertEqual(loss, {"L0", "chinchilla3", "chinchilla_interaction", "width_depth"})
        self.assertEqual(hp, {"steplaw_lr", "steplaw_bsz"})

    def test_each_form_is_well_formed(self):
        for name, form in PARAMETRIC_FORMS.items():
            with self.subTest(form=name):
                self.assertIsInstance(form, ParametricForm)
                self.assertTrue(form.input_cols)
                self.assertTrue(callable(form.func))
                self.assertTrue(callable(form.interpretable))
                self.assertTrue(callable(form.format_fit))
                self.assertIsInstance(form.signed_params, frozenset)

    def test_l0_interpretable_and_curve(self):
        form = PARAMETRIC_FORMS["L0"]
        params = np.array([np.log(1.0), np.log(200.0), -0.1])
        interp = form.interpretable(params)
        self.assertEqual(set(interp), {"L0", "A", "alpha"})
        self.assertIsInstance(form.format_fit(params), str)
        L = form.to_L_vs_C(np.array([1e12, 1e15]), params)
        self.assertEqual(len(L), 2)
        self.assertTrue(np.all(np.isfinite(L)))


class TestSurrogateRegistry(unittest.TestCase):
    def test_expected_models_present(self):
        self.assertEqual(
            set(MODELS),
            {"none", "gp", "tabpfn_ens", "tabpfn_bo", "ag", "xgb_ens"},
        )

    def test_null_surrogate_predicts_nan(self):
        # model-free path: fit is a no-op, predict returns NaN so the caller's dropna keeps only the
        # observed points. Predicting NaN of the right length is the whole contract.
        model = NullSurrogate()
        model.fit(np.zeros((3, 2)), np.zeros((3, 1)))
        pred = model.predict(np.zeros((4, 2)))
        self.assertEqual(len(pred), 4)
        self.assertTrue(np.all(np.isnan(pred)))


class TestSelectionRegistry(unittest.TestCase):
    def test_expected_selectors_present(self):
        self.assertEqual(
            set(SELECTION),
            {"random", "local_search", "max_variance", "farthest_point",
             "ei", "ei_per_cost", "tabpfn_ei"},
        )

    def test_surrogate_free_selectors_instantiate(self):
        # random / farthest_point must work without a GP (they drive the model-free baseline).
        for name in ("random", "farthest_point"):
            with self.subTest(selection=name):
                sel = SELECTION[name](seed=0)
                self.assertFalse(getattr(sel, "needs_surrogate", False))


if __name__ == "__main__":
    unittest.main()
