"""End-to-end smoke test: ``main.py`` runs both tasks and writes a usable history.

Runs the model-free path (``--model_type none``) so no GP/torch backend is needed -- this exercises
the full dispatch -> prepare_data -> acquisition loop -> fit -> save chain, which is exactly what a
reorg is most likely to break. Kept to a few iterations on the tiny fixture grid.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests._grid import REPO_ROOT, write_grid

# shared flags: model-free + random so the run needs no surrogate; wide window so the whole grid is
# the sampling pool; few points so it finishes in seconds.
_COMMON = [
    "--seed", "0", "--niters", "2", "--init_points", "6",
    "--model_type", "none", "--selection", "random",
    "--init_compute_factor", "1e6", "--C_window", "1e-6", "1e6",
]


class TestPipelinesSmoke(unittest.TestCase):
    def _run(self, task, form):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            grid = write_grid(tmp / "grid.csv")
            out = tmp / "out"
            cmd = [sys.executable, "main.py", task,
                   "--input", str(grid), "--output_dir", str(out), "--form", form, *_COMMON]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"{task} failed:\n{proc.stderr[-2000:]}")

            seed_dir = out / "history" / "grid" / form / "seed_0"
            self.assertTrue(seed_dir.is_dir(), f"{task} wrote no seed dir")
            run_dir = max(seed_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            hist_path = run_dir / "history.parquet"
            args_path = run_dir / "args.json"
            self.assertTrue(hist_path.exists(), f"{task} wrote no history parquet")
            self.assertTrue(args_path.exists(), f"{task} wrote no args.json")
            hist = pd.read_parquet(hist_path)
            self.assertGreater(len(hist), 0)
            # every acquisition row carries the two-region eval scores + the fitted params
            self.assertIn("law_mse_pool", hist.columns)
            self.assertIn("law_mse_held_out", hist.columns)
            self.assertIn("known_test_mse", hist.columns)  # back-compat alias of held-out
            self.assertIn("params", hist.columns)
            return hist

    def test_loss_law_l0(self):
        hist = self._run("loss_law", "L0")
        self.assertIn("regret_alpha", hist.columns)  # L0's headline coefficient

    def test_hp_law_steplaw_lr(self):
        hist = self._run("hp_law", "steplaw_lr")
        self.assertIn("regret_coef", hist.columns)  # step-law coefficient


if __name__ == "__main__":
    unittest.main()
