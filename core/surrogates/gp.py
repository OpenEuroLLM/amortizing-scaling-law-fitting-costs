import numpy as np

from .base import SurrogateModel


class GPModel(SurrogateModel):
    def __init__(self, **kwargs):
        self._gp = None
        self._likelihood = None
        self.use_mean_model = False
        if "use_mean_model" in kwargs:
            self.use_mean_model = kwargs["use_mean_model"]
        self._mean_kwargs = {}
        if "mean_kwargs" in kwargs:
            self._mean_kwargs.update(kwargs["mean_kwargs"])

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, max_attempts: int = 10, **kwargs) -> None:
        # lazy import to avoid requiring gpytorch/botorch unless GPModel is used
        import botorch, gpytorch, torch
        from .means import ScalingLawMean_L0_form as MeanL0

        self._gp = botorch.models.SingleTaskGP(
            torch.tensor(train_x),
            torch.tensor(train_y),
            mean_module=MeanL0(**self._mean_kwargs) if self.use_mean_model else None,
            outcome_transform=None,
            covar_module=gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1]),
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._gp.likelihood, self._gp)
        botorch.fit_gpytorch_mll(mll, max_attempts=max_attempts)

        if kwargs.get("diagnose", False):
            with torch.no_grad():
                self._gp.train()
                mll_val = mll(self._gp(*self._gp.train_inputs), self._gp.train_targets).item()
                train_pred = self._gp.posterior(self._gp.train_inputs[0]).mean
                train_mse = ((train_pred - self._gp.train_targets.unsqueeze(-1)) ** 2).mean().item()
            _n = self._gp.train_targets.numel()
            print(f"  [GP] n={_n}  MLL/n={mll_val:.3f}  train_mse={train_mse:.2e}")
            print(f"  [GP] n={_n} noise={self._gp.likelihood.noise.item():.2e}")
            print(f"  [GP] n={_n} lengthscales={self._gp.covar_module.lengthscale.detach().numpy().ravel().round(2)}")  # N,D,lr,gbsz
            print(f"  [GP] n={_n} mean={ {n: round(p.item(), 3) for n, p in self._gp.mean_module.named_parameters()} }")

        self._gp.eval()

    def predict(self, X_test: np.ndarray, return_std: bool = False):
        import torch

        with torch.no_grad():
            post = self._gp.posterior(torch.tensor(X_test))
            mean = post.mean.numpy().squeeze(-1)
            if return_std:
                std = post.variance.clamp_min(1e-12).sqrt().numpy().squeeze(-1)
                return mean, std
            return mean
