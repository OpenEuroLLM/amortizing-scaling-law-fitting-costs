from abc import ABC, abstractmethod

import numpy as np


class SurrogateModel(ABC):
    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """train_x and train_y are already log-transformed."""

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """X_test is already log-transformed. Returns 1-D predictions."""


class GPModel(SurrogateModel):
    def fit(self, train_x, train_y):
        import botorch
        import torch
        import gpytorch

        self._gp = botorch.models.SingleTaskGP(
            torch.tensor(train_x),
            torch.tensor(train_y),
            outcome_transform=None,
            covar_module=gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1]),
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._gp.likelihood, self._gp)
        botorch.fit_gpytorch_mll(mll, max_attempts=3)
        self._gp.eval()
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def predict(self, X_test):
        import torch

        with torch.no_grad():
            return self._likelihood(self._gp(torch.tensor(X_test))).mean.numpy()


class TabPFNModel(SurrogateModel):
    def fit(self, train_x, train_y):
        from tabpfn import TabPFNRegressor

        self._reg = TabPFNRegressor()
        self._reg.fit(train_x, train_y)

    def predict(self, X_test):
        return self._reg.predict(X_test)


class AutoGluonModel(SurrogateModel):
    def fit(self, train_x, train_y):
        import pandas as pd
        from autogluon.tabular import TabularPredictor

        train_data = pd.DataFrame(train_x, columns=[f"f{i}" for i in range(train_x.shape[1])])
        train_data["labels"] = train_y.ravel()
        self._predictor = TabularPredictor(label="labels").fit(train_data)
        self._feature_names = train_data.columns.drop("labels").tolist()

    def predict(self, X_test):
        import pandas as pd

        X_df = pd.DataFrame(X_test, columns=self._feature_names)
        return self._predictor.predict(X_df).to_numpy()


MODELS = {
    "gp": GPModel,
    "tabpfn": TabPFNModel,
    "ag": AutoGluonModel,
}
