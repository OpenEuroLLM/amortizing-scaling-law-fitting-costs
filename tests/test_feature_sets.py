"""``resolve_grid_features`` selects the right GP feature columns and drops signal-free ones.

The auto-narrowing (drop absent / all-null / constant columns) is what lets one --feature_set flag
work across datasets with different knobs; these pin that behavior.
"""
import unittest

import numpy as np

from core.utils.data import resolve_grid_features
from tests._grid import make_grid


class TestResolveGridFeatures(unittest.TestCase):
    def setUp(self):
        self.df = make_grid()  # varies N, D, lr, gbsz

    def test_default_is_the_four_config_axes(self):
        self.assertEqual(resolve_grid_features(self.df, "default"), ["N", "D", "lr", "gbsz"])

    def test_all_null_column_is_dropped(self):
        df = self.df.copy()
        df["lr"] = np.nan  # an HP-less source: lr present but entirely missing
        self.assertEqual(resolve_grid_features(df, "default"), ["N", "D", "gbsz"])

    def test_constant_column_is_dropped(self):
        df = self.df.copy()
        df["gbsz"] = 256  # a study that fixes the batch size
        self.assertEqual(resolve_grid_features(df, "default"), ["N", "D", "lr"])

    def test_arch_superset_keeps_n_and_d(self):
        df = self.df.copy()
        df["width"] = np.tile([128, 256, 512], len(df) // 3 + 1)[: len(df)]
        default = resolve_grid_features(df, "default")
        arch = resolve_grid_features(df, "arch")
        self.assertIn("width", arch)
        self.assertTrue(set(default).issubset(arch))  # full superset: N/D never replaced
        self.assertNotIn("width", default)

    def test_absent_arch_columns_collapse_to_default(self):
        # no width/depth/... columns present -> arch resolves to exactly default
        self.assertEqual(
            resolve_grid_features(self.df, "arch"), resolve_grid_features(self.df, "default")
        )

    def test_unknown_feature_set_raises(self):
        with self.assertRaises(ValueError):
            resolve_grid_features(self.df, "bogus")


if __name__ == "__main__":
    unittest.main()
