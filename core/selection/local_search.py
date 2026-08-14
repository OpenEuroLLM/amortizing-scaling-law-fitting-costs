import numpy as np
import pandas as pd

from .base import DataSelection, TARGET


class LocalSearch(DataSelection):
    def __init__(self, seed: int, starting_point: dict = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.starting_point = starting_point

    def split(self, df_fit, n_points):

        grid_values = []
        loss_values = []
        optimum_per_scale = {}
        total_compute = 0

        n_scale = np.sort(df_fit['N'].unique())
        d_scale = np.sort(df_fit['D'].unique())

        # start on the smallest N, D
        current_scale_model_size = n_scale.min()
        current_scale_token_budget = d_scale.min()

        if self.starting_point is None:
            # sample a random point on the smallest scale
            lr_grid = df_fit[(df_fit['N'] == current_scale_model_size)&(df_fit['D'] == current_scale_token_budget)]['lr'].unique()
            gbsz_grid = df_fit[(df_fit['N'] == current_scale_model_size)&(df_fit['D'] == current_scale_token_budget)]['gbsz'].unique()
            incumbent = {'lr': self.rng.choice(lr_grid), 'gbsz': self.rng.choice(gbsz_grid)}
        else:
            incumbent = self.starting_point
        current_best_observed_loss = None
        reset_on_new_scale = True

        while len(grid_values) < n_points:

            # if we are on a new scale, we start with evaluating the incumbent
            if reset_on_new_scale:
                # evaluate current point
                index_incumbent = df_fit.index[
                    (df_fit['N'] == current_scale_model_size) & (df_fit['D'] == current_scale_token_budget) & (
                                df_fit['lr'] == incumbent['lr']) & (df_fit['gbsz'] == incumbent['gbsz'])]

                # FIXME: if we don't have observed this point, continue with a random configuration, this can only happen for incomplete grids
                if len(index_incumbent) == 0:
                    lr_grid = \
                    df_fit[(df_fit['N'] == current_scale_model_size) & (df_fit['D'] == current_scale_token_budget)][
                        'lr'].unique()
                    gbsz_grid = \
                    df_fit[(df_fit['N'] == current_scale_model_size) & (df_fit['D'] == current_scale_token_budget)][
                        'gbsz'].unique()
                    incumbent = {'lr': self.rng.choice(lr_grid), 'gbsz': self.rng.choice(gbsz_grid)}
                    index_incumbent = df_fit.index[(df_fit['N'] == current_scale_model_size) & (df_fit['D'] == current_scale_token_budget) & (
                            df_fit['lr'] == incumbent['lr']) & (df_fit['gbsz'] == incumbent['gbsz'])]

                loss_value = df_fit[TARGET].iloc[index_incumbent].values[0]

                # bookkeeping
                config = (current_scale_model_size, current_scale_token_budget, incumbent['lr'],
                          incumbent['gbsz'])
                grid_values.append(config)
                loss_values.append(loss_value)
                total_compute += df_fit['C'].iloc[index_incumbent].values[0]
                current_best_observed_loss = loss_value
                reset_on_new_scale = False

            # otherwise we continue with evaluating the neighbors on the current scale
            else:
                ## get the cross neighbors
                neighbours = []
                lr_grid = np.sort(df_fit[(df_fit['N'] == current_scale_model_size)&(df_fit['D'] == current_scale_token_budget)]['lr'].unique())

                lower = list(lr_grid).index(incumbent['lr']) - 1
                upper = list(lr_grid).index(incumbent['lr']) + 1

                if lower >= 0:
                    neighbours.append({'lr': lr_grid[lower], 'gbsz': incumbent['gbsz']})

                if upper < len(lr_grid):
                    neighbours.append({'lr': lr_grid[upper], 'gbsz': incumbent['gbsz']})

                gbsz_grid = np.sort(df_fit[(df_fit['N'] == current_scale_model_size)&(df_fit['D'] == current_scale_token_budget)]['gbsz'].unique())

                try:
                    lower = list(gbsz_grid).index(incumbent['gbsz']) - 1
                    upper = list(gbsz_grid).index(incumbent['gbsz']) + 1
                except ValueError:
                    print(gbsz_grid, incumbent)
                if lower >= 0:
                    neighbours.append({'lr': incumbent['lr'], 'gbsz': gbsz_grid[lower]})

                if upper < len(gbsz_grid):
                    neighbours.append({'lr': incumbent['lr'], 'gbsz': gbsz_grid[upper]})

                ## evaluate all neighbors, stop the evaluation if we found something better
                found_something_better = False
                for neighbour in neighbours:

                    # have we seen this point already?
                    config = (current_scale_model_size, current_scale_token_budget, neighbour['lr'],
                              neighbour['gbsz'])
                    if config in grid_values:
                        continue

                    # evaluate
                    index_incumbent = df_fit.index[
                        (df_fit['N'] == current_scale_model_size) & (df_fit['D'] == current_scale_token_budget) & (
                                    df_fit['lr'] == neighbour['lr']) & (df_fit['gbsz'] == neighbour['gbsz'])]

                    if len(index_incumbent) == 0:
                        continue
                    loss_value = df_fit[TARGET].iloc[index_incumbent].values[0]

                    # bookkeeping
                    grid_values.append(config)
                    loss_values.append(loss_value)
                    total_compute += df_fit['C'].iloc[index_incumbent].values[0]

                    ## if we have improved, we can stop here
                    if current_best_observed_loss > loss_value:
                        current_best_observed_loss = loss_value
                        incumbent = neighbour
                        found_something_better = True
                        break

                ## if we haven't found anything better, we will go to the next scale
                if found_something_better is False:
                    reset_on_new_scale = True

                    # save the optimum at this scale
                    if current_scale_model_size not in optimum_per_scale:
                        optimum_per_scale[current_scale_model_size] = {}
                    optimum_per_scale[current_scale_model_size][current_scale_token_budget] = incumbent

                    ## check if we are already at the highest token budget, if not increase D
                    if current_scale_token_budget < d_scale.max():
                        index_current_token_budget = list(d_scale).index(current_scale_token_budget)
                        # go to the next highest token budget
                        current_scale_token_budget = list(d_scale)[index_current_token_budget + 1]
                    ## if we are already at the highest token budget, we increase model scale N
                    else:

                        # reset D <- D_min
                        current_scale_token_budget = d_scale.min()

                        # reset the incumbent to the optimal config we found on this scale on the D_min
                        incumbent = optimum_per_scale[current_scale_model_size][current_scale_token_budget]

                        # increase N
                        index_current_model_scale = list(n_scale).index(current_scale_model_size)
                        current_scale_model_size = list(n_scale)[index_current_model_scale + 1]



        return np.array(grid_values), np.array(loss_values), total_compute, None

    def next_point(
        self,
        df: pd.DataFrame,
        candidate_set: pd.Index | None = None,
        replace: bool = False,
        **kwargs
    ) -> int:
        raise NotImplementedError("LocalSearch does not support next_point selection yet.")
