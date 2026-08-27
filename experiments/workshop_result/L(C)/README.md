# L(C) — compute-only recovery

BO with **Expected Improvement** (`--selection ei`) on `oellm_english`, form **`L0`**
(`L(C) = L0 + A·C^α`), GP surrogate, practice-style init, run to **full convergence** (the whole
acquirable pool is acquired), **10 seeds** per arm.

Two arms differ only in the acquisition compute window `[lo·C_lim, hi·C_lim]` (`C_lim` = max compute
acquired so far):

| arm | filename | `--C_window` | meaning |
|-----|----------|--------------|---------|
| **full**   | `bo_ei_full` | `1e-5 1e3` | ceiling clears the whole pool every step → EI picks from all remaining points |
| **window** | `bo_ei`      | `1e-5 1.5` | ceiling capped at `1.5·C_lim` → grows with the frontier |

## Reproduce

```bash
bash "experiments/workshop_result/L(C)/run.sh"
```

Writes to `experiments/workshop_result/L(C)/runs/` (gitignored; each seed ≈ 40 min CPU). The shared
analysis notebook (`../analysis/recovery_grid.ipynb`) reads these runs as the **top** row of the grid.
