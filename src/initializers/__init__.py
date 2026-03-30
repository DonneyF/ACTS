"""
Initializer module for optimization. All initializers return data in [0,1]^d
"""

import logging
import math
import urllib
from pathlib import Path

import pandas as pd
import torch
from botorch.utils.sampling import sample_hypersphere
from scipy.stats import vonmises_fisher
from torch import Tensor


def sobol(n: int, d: int, test_function, seed: int = None, radius: float = None, **kwargs) -> Tensor:
    """Sobol initialization"""
    lb = test_function.bounds[0]
    ub = test_function.bounds[1]
    if not radius:
        sobol_engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        return sobol_engine.draw(n).to(ub)
    else:
        return (sample_hypersphere(d=d, n=n, qmc=True, seed=seed).to(lb) * radius - lb) / (ub - lb)


def gaussian(n: int, d: int, test_function, seed: int = None, **kwargs) -> Tensor:
    """Gaussian initialization"""
    lb = test_function.bounds[0]
    ub = test_function.bounds[1]
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    X_init = torch.randn(n, d, generator=gen).to(ub) / math.sqrt(d)
    # X_init /= X_init.norm(dim=-1).max() # Each element has norm <= 1 with at least one point = 1.
    return (X_init - lb) / (ub - lb)


__all__ = ["sobol", "gaussian"]
