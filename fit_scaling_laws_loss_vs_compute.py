import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import seaborn as sns
from plot_utils import N_palette, D_palette, Ns, Ds, N_color_map, D_color_map, FORMAT_BILLIONS, N_ticks, D_ticks
from scipy.optimize import curve_fit


loss_key = 'loss'

df = pd.read_csv('data/oellm_english/preprocessed/oellm_english.csv')
df.head()

# Any duplicate runs
cols = ['N','D','gbsz','lr']
df[df.duplicated(cols, keep=False)].sort_values(['N','D','lr','gbsz']).head()
df= df[df['N'] != 1.7e9]

# Add compute
df["C"] = 6 * df["N"] * df["D"]

# Optimal (lr, gbsz) for each (N, D)
df_best_lr_gbsz = df.loc[df.groupby(["N", "D"])[loss_key].idxmin()].reset_index(drop=True)
df_best_lr_gbsz = df_best_lr_gbsz[df_best_lr_gbsz.N != 1.7e9]
N_color_map = dict(zip(sorted(df_best_lr_gbsz["N"].unique()), N_palette))

# Extract Pareto points of L vs C.
df_env = df_best_lr_gbsz.sort_values(["C", loss_key])

rows = []
best_loss = float("inf")

for _, row in df_env.iterrows():
    if row[loss_key] < best_loss:
        rows.append(row)
        best_loss = row[loss_key]

df_env = pd.DataFrame(rows).reset_index(drop=True)


# ============================================================
# pure power-law fit
# ============================================================

def power_law(C, A, alpha):
    return A * C**(-alpha)

params_env, _ = curve_fit(
    power_law,
    df_env["C"].values,
    df_env[loss_key].values,
    p0=[1e2, 0.1],
)

A_fit, alpha_fit = params_env

print(f"L(C) = {A_fit:.3e} * C^(-{alpha_fit:.4f})")

# ============================================================
# prediction curve
# ============================================================

C_grid = np.logspace(
    np.log10(df_env["C"].min()),
    np.log10(df_env["C"].max()),
    300,
)

L_grid = power_law(
    C_grid,
    *params_env,
)

# ============================================================
# plot
# ============================================================

fig, ax = plt.subplots(figsize=(6, 4))


# all runs
sns.scatterplot(
    data=df_best_lr_gbsz,
    x="C",
    y=loss_key,
    hue="N",
    alpha=0.3,
    palette=N_color_map,
    edgecolor="black",
    s=30,
    ax=ax,
    legend=False,
)

# envelope points
sns.scatterplot(
    data=df_env,
    x="C",
    y=loss_key,
    hue="N",
    palette=N_color_map,
    s=30,
    edgecolor="black",
    linewidth=0.5,
    ax=ax,
    legend=True,
    zorder=5,
)

# fitted curve
ax.plot(
    C_grid,
    L_grid,
    color="red",
    linewidth=1.5,
    alpha=0.7,
    linestyle="--",
    label=rf"$L(C)={A_fit:.1f} C^{{-{alpha_fit:.3f}}}$",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"FLOPs")
ax.set_ylabel(r"$\mathcal{L}_{\mathrm{valid}}$")

ax.set_title("Loss vs Compute, with Fitted Power Law on the Loss Envelope")

handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles,
    [
        FORMAT_BILLIONS(float(label))
        if label not in [rf"$L(C)={A_fit:.1f} C^{{-{alpha_fit:.3f}}}$"]
        else label
        for label in labels
    ],
    title=r"$N$",
    loc="lower left",
    ncol=2,
    fontsize=9
)

plt.tight_layout()
Path('results').mkdir(exist_ok=True)
plt.savefig('results/fit_compute.png')
plt.show()