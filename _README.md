To install the environment using [uv](https://docs.astral.sh/uv/getting-started/installation/).
```bash
# with the presence of `uv.lock`
uv sync --extra gp
```

Basic test run:
```bash
# assuming the data folder and the .csv exists
python iterative_fits_L0_form.py \
--seed 12345 \
--niters 25 \
--output_dir results/iterative/ \
--input data/all_points_clean.csv \
--init_points 25 \
--init_compute_factor 50 \
--C_window 0.1 50 \
--plot_figure
```