from typing import List
import numpy as np
import torch
import gpytorch


class ScalingLawMean_L0_form(gpytorch.means.Mean):

    def __init__(
        self, 
        init_params: List[float] = (0.2, 3.5, -0.5),
        col_N: int = 0,
        col_D: int = 1,
        **kwargs
    ):
        super().__init__()  # log L0, log A, b
        # registering to ensure they are optimized during ML training
        self.register_parameter("l0", torch.nn.Parameter(torch.tensor(float(init_params[0]), dtype=torch.float64)))
        self.register_parameter("a",  torch.nn.Parameter(torch.tensor(float(init_params[1]), dtype=torch.float64)))
        self.register_parameter("b",  torch.nn.Parameter(torch.tensor(float(init_params[2]), dtype=torch.float64)))
        self.col_N = col_N
        self.col_D = col_D

    def forward(self, x):
        logC = torch.log(torch.tensor(6.0, dtype=torch.float64)) + x[..., self.col_N] + x[..., self.col_D]        # cols 0,1 = log N, log D
        return self.l0.exp() + torch.exp(self.a + self.b * logC)
# end of file