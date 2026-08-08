import os
import json

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from tueplots import bundles

# Plug any of those into either the rcParams or into an rc_context:
plt.rcParams.update(bundles.neurips2024())

points = [50, 100, 200, 500]
seeds = 5
max_compute = 2.0522423999999998e+23

models = ['gp', 'ag']
selections = ['random', 'local_search']
data_folder = Path('../output')

fig, ax = plt.subplots(2, sharex=True)

colors = ['C0', 'C1']
alpha = [0.4, 0.6, 0.8, 1.0]

for k, model in enumerate(models):
    for l, selection in enumerate(selections):
        plot_legend = True
        for i, num_points in enumerate(points):
            A = []
            B = []
            compute = []
            for j, seed in enumerate(range(seeds)):

                fname = f'results_{model}_{selection}_{num_points}_points_seed_{seed}.json'
                if not os.path.exists(data_folder / fname):
                    continue
                res = json.load(open(data_folder / fname))
                A.append(res['A_fit'])
                B.append(res['alpha_fit'])
                compute.append(res['total_compute'])

            if len(compute) == 0:
                continue

            if plot_legend:
                ax[0].errorbar(np.mean(compute), np.mean(A), xerr=np.std(compute), yerr=np.std(A), fmt='x',
                               color=colors[l], alpha=alpha[i], label=selection)
                ax[1].errorbar(np.mean(compute), np.mean(B), xerr=np.std(compute), yerr=np.std(B), fmt='x',
                               color=colors[l], alpha=alpha[i], label=selection)
                plot_legend = False
            else:
                ax[0].errorbar(np.mean(compute), np.mean(A), xerr=np.std(compute), yerr=np.std(A), fmt='x', color=colors[l], alpha=alpha[i])
                ax[1].errorbar(np.mean(compute), np.mean(B), xerr=np.std(compute), yerr=np.std(B), fmt='x', color=colors[l], alpha=alpha[i])

ax[0].scatter(x= max_compute, y= 2.482e+01, color='r', marker='D')
ax[0].set_ylabel(r'$A$')

ax[0].legend()
ax[0].grid()

ax[1].legend()
ax[1].grid()
ax[1].scatter(x= max_compute, y=0.0513, color='r', marker='D')
ax[1].set_ylabel(r'$\alpha$')
ax[1].set_xlabel('total compute')
ax[1].set_xscale('log')
plt.savefig('fits.pdf')

#
#plt.figure(dpi=200)
#for k, model in enumerate(models):
#    plt.errorbar(np.arange(len(subsets)), B[k].mean(axis=-1), yerr=A.std(axis=-1), fmt='o', color=f'C{k}', label=model)
#plt.axhline(y=0.0513, color='k', linestyle='--')
#plt.xticks(np.arange(len(subsets)), ticks)
#plt.ylabel(r'$\alpha$')
#plt.xlabel('subset')
#plt.legend()
#plt.grid()
#plt.savefig('alpha.png')
#
#
#
#plt.figure(dpi=200)
#for k, model in enumerate(models):
#    plt.scatter(compute[k] / max_compute, A[k], marker='x', c=f'C{k}', s=20, label=model)
#plt.scatter(1, y= 2.482e+01, color='r', marker='o')
#plt.ylabel(r'$A$')
#plt.xlabel('relative compute')
#
#plt.legend()
#plt.grid()
#plt.savefig('A_vs_compute.png')
#
#
#plt.figure(dpi=200)
#for k, model in enumerate(models):
#    plt.scatter(compute[k] / max_compute, B[k], marker='x', c=f'C{k}', s=20, label=model)
#plt.scatter(1, y=0.0513, color='r', marker='o')
#plt.ylabel(r'$\alpha$')
#plt.xlabel('relative compute')
#
#plt.legend()
#plt.grid()
#plt.savefig('alpha_vs_compute.png')