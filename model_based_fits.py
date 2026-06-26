import argparse
import json

import numpy as np
import pandas as pd

from pathlib import Path
from scipy.optimize import curve_fit

from models import MODELS
from selection import SELECTION

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--num_points', type=int, required=True)
parser.add_argument('--model_type', type=str, default='gp', choices=list(MODELS))
parser.add_argument('--selection', type=str, default='random', choices=list(SELECTION))
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input)

df["C"] = 6 * df["N"] * df["D"]
df = df[df['N'] != 1.7e9]
df_fit = pd.DataFrame(df).reset_index(drop=True)

# Generate training data
n_points = args.num_points
splitter = SELECTION[args.selection](seed=args.seed)
X, y, total_compute = splitter.split(df_fit, n_points)

train_x = np.log(X)
train_y = y[:, None]

X_test = np.log(df_fit[['N', 'D', 'lr', 'gbsz']].to_numpy())

# Fit and predict
model = MODELS[args.model_type]()
model.fit(train_x, train_y)
Y_test = model.predict(X_test)

df_pred = df_fit.copy()
df_pred['loss'] = Y_test

# Optimal (lr, gbsz) for each (N, D)
df_best_lr_gbsz = df_pred.loc[df_pred.groupby(["N", "D"])['loss'].idxmin()].reset_index(drop=True)
df_best_lr_gbsz = df_best_lr_gbsz[df_best_lr_gbsz.N != 1.7e9]

# Extract Pareto points of L vs C
df_env = df_best_lr_gbsz.sort_values(["C", 'loss'])
rows = []
best_loss = float("inf")
for _, row in df_env.iterrows():
    if row['loss'] < best_loss:
        rows.append(row)
        best_loss = row['loss']
df_env = pd.DataFrame(rows).reset_index(drop=True)

mse = np.mean((Y_test - df_fit['loss']) ** 2)

# Pure power-law fit
def power_law(C, A, alpha):
    return A * C ** (-alpha)

params_env, _ = curve_fit(power_law, df_env["C"].values, df_env['loss'].values, p0=[1e2, 0.1])
A_fit, alpha_fit = params_env

print(f'MSE = {mse}')
print(f"L(C) = {A_fit:.3e} * C^(-{alpha_fit:.4f})")

results = {
    'A_fit': A_fit,
    'alpha_fit': alpha_fit,
    'seed': args.seed,
    'n_points': n_points,
    'mse': float(mse),
    'total_compute': total_compute,
}
json.dump(results, open(Path(args.output_dir) / f'results_{args.model_type}_{args.selection}_{n_points}_points_seed_{args.seed}.json', 'w'))
