from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import pandas as pd

FEATURES = ['N', 'D', 'lr', 'gbsz']
TARGET = 'loss'


class DataSelection(ABC):
    @abstractmethod
    def split(self, df_fit: pd.DataFrame, n_points: int) -> tuple[pd.DataFrame, pd.Series]:
        """Return (X, y) with n_points training rows selected from df_fit."""


class RandomSelection(DataSelection):
    def __init__(self, seed: int):
        self.seed = seed

    def split(self, df_fit, n_points):
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(df_fit.index, n_points, replace=False)
        return df_fit[FEATURES].iloc[idx].to_numpy(), df_fit[TARGET].iloc[idx].to_numpy(), df_fit['C'].iloc[idx].sum()

class LocalSearch(DataSelection):
    def __init__(self, seed: int, starting_point: dict = None):
        self.seed = seed
        self.starting_point = starting_point

    def split(self, df_fit, n_points):

        grid_values = []
        loss_values = []
        total_compute = 0

        n_scale = df_fit['N'].unique()
        d_scale = df_fit['D'].unique()

        current_scale_model_size = n_scale.min()
        current_scale_token_budget = d_scale.min()

        if self.starting_point is None:
            # sample a random point on the smallest scale
            lr_grid = df_fit[(df_fit['N'] == current_scale_model_size)&(df_fit['D'] == current_scale_token_budget)]['lr'].unique()
            gbsz_grid = df_fit[(df_fit['N'] == current_scale_model_size)&(df_fit['D'] == current_scale_token_budget)]['gbsz'].unique()
            self.starting_point = {'lr': np.random.choice(lr_grid), 'gbsz': np.random.choice(gbsz_grid)}

        current_best = df_fit.index[(df_fit['N'] == current_scale_model_size)&(df_fit['D'] == current_scale_token_budget)&(df_fit['lr'] == self.starting_point['lr'])&(df_fit['gbsz'] == self.starting_point['gbsz'])]
        loss_value = df_fit[TARGET].iloc[current_best].values[0]
        total_compute += df_fit['C'].iloc[current_best].values[0]
        config = (current_scale_model_size, current_scale_token_budget, self.starting_point['lr'], self.starting_point['gbsz'])
        grid_values.append(config)
        loss_values.append(loss_value)


        return np.array(grid_values), np.array(loss_values), total_compute


SELECTION = {
    "random": RandomSelection,
    "local_search": LocalSearch,
}
