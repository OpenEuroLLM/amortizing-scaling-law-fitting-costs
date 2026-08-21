"""``split_search_space`` carves the pool + held-out regions with the right sizes and ordering."""
import unittest

from core.utils.data import split_search_space
from tests._grid import make_grid


class TestSplitSearchSpace(unittest.TestCase):
    def setUp(self):
        self.df = make_grid()  # 135 rows across N x D x lr x gbsz

    def test_two_disjoint_regions_cover_the_grid(self):
        pool, held = split_search_space(self.df, held_out_frac=0.2)
        self.assertFalse(set(pool.index) & set(held.index))          # disjoint
        self.assertEqual(set(pool.index) | set(held.index), set(self.df.index))  # cover

    def test_held_out_is_the_largest_compute(self):
        pool, held = split_search_space(self.df, held_out_frac=0.2)
        self.assertGreaterEqual(held["C"].min(), pool["C"].max())    # top slice by compute

    def test_larger_frac_holds_out_more(self):
        # held_out_frac is a threshold on the axis value (not a row quantile), so held-out size
        # isn't proportional to the fraction -- but it must grow monotonically with it.
        _, held_small = split_search_space(self.df, held_out_frac=0.2)
        _, held_large = split_search_space(self.df, held_out_frac=0.8)
        self.assertGreaterEqual(len(held_large), len(held_small))

    def test_axis_N_and_bad_args_raise(self):
        _, held = split_search_space(self.df, held_out_frac=0.2, axis="N")
        self.assertEqual(held["N"].min(), self.df["N"].max())        # only the largest N slice
        for bad in ({"axis": "lr"}, {"held_out_frac": 0}, {"held_out_frac": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                split_search_space(self.df, **bad)


if __name__ == "__main__":
    unittest.main()
