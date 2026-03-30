"""
Expected Improvement acquisition functions.
"""

import functools
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Optional

import numpy as np
import numpy.typing as npt
import pymanopt as pt
import pymanopt.autodiff.backends
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition import LogExpectedImprovement as LogEI
from botorch.acquisition import LogNoisyExpectedImprovement as LogNoisyEI
from botorch.acquisition import qLogExpectedImprovement as qLogEI
from botorch.acquisition import qLogNoisyExpectedImprovement as qLogNoisyEI
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _log_ei_helper, _scaled_improvement
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import (
    BadInitialCandidatesWarning,
    BotorchTensorDimensionError,
    BotorchWarning,
    SamplingWarning,
)
from botorch.exceptions.errors import OptimizationGradientError
from botorch.generation import gen_candidates_scipy, gen_candidates_torch
from botorch.generation.gen import _process_scipy_result
from botorch.models.model import Model
from botorch.optim import (
    initialize_q_batch,
    initialize_q_batch_nonneg,
    initialize_q_batch_topn,
    optimize_acqf,
)
from botorch.optim.initializers import is_nonnegative
from botorch.optim.parameter_constraints import (
    _arrayify,
    make_scipy_bounds,
)
from botorch.optim.utils import columnwise_clamp, fix_features, get_X_baseline, minimize_with_timeout
from botorch.utils import average_over_ensemble_models, t_batch_mode_transform
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.probability.utils import log_ndtr as log_Phi  # noqa: N812
from botorch.utils.probability.utils import log_phi as log_phi
from botorch.utils.sampling import sample_hypersphere
from jaxtyping import Float
from pymanopt.autodiff.backends._backend import Backend
from pymanopt.optimizers.optimizer import Optimizer as PymanoptOptimizer
from scipy.optimize import minimize, root_scalar
from torch import Tensor
from torch.quasirandom import SobolEngine

from src.acquisitions.base import Acquisition


def sym_cube_to_unit_cube(x: torch.Tensor) -> torch.Tensor:
    """Maps data from [-1, 1]^d to [0, 1]^d"""
    return (x + 1) / 2


def unit_cube_to_sym_cube(x: torch.Tensor) -> torch.Tensor:
    """Maps data from [0, 1]^d to [-1,1]^d"""
    return 2 * x - 1


@dataclass
class LogExpectedImprovement(Acquisition):
    """
    Generic acquisition for analytic Log Expected Improvement.
    Only supports q=1
    By default optimizes on same device as X, optionally falls back to CPU.
    """

    gen_candidates_func: Literal["torch", "scipy"] = "scipy"
    optimize_device: torch.device = None  # Defaults to same device as X
    cpu_fallback: bool = False
    noisy: bool = False

    def __repr__(self):  # noqa: D105
        return "LogEI" if not self.noisy else "LogNoisyEI"

    def argmax(
        self,
        model,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
        q: int,
        bounds: Float[Tensor, "2 d"],
        options: Optional[dict] = None,
        **kwargs,
    ) -> Float[Tensor, "q d"]:
        """
        Computes the analytic Log Expected Improvement.
        """
        options = options or {}
        if q != 1:
            raise ValueError("LogEI: q must be 1")

        optimize_device = X.device if self.optimize_device is None else self.optimize_device

        while True:
            if self.noisy:
                LogEI_acqf = LogNoisyEI(model.to(optimize_device), X_observed=X)
            else:
                LogEI_acqf = LogEI(model.to(optimize_device), Y.max())

            try:
                bounds_ = bounds.to(optimize_device)
                X_next, values = optimize_acqf(
                    LogEI_acqf,
                    bounds=bounds_,
                    q=1,
                    gen_candidates=gen_candidates_torch
                    if self.gen_candidates_func == "torch"
                    else gen_candidates_scipy,
                    **options.get("optimize_acqf"),
                )
                break
            except torch.OutOfMemoryError as e:
                if self.cpu_fallback and self.optimize_device != torch.device("cpu"):
                    model._clear_cache()
                    self.optimize_device = torch.device("cpu")
                else:
                    raise e

        X_next = X_next.detach().to(X)

        return X_next


@dataclass
class qLogExpectedImprovement(Acquisition):  # noqa: N801
    """
    Generic acquisition for MCMC q- Log Expected Improvement.
    By default optimizes on same device as X, optionally falls back to CPU.
    """

    sequential: bool = False  # Sequential optimization to build the batch
    gen_candidates_func: Literal["torch", "scipy"] = "scipy"
    optimize_device: torch.device = None  # Defaults to same device as X
    cpu_fallback: bool = False
    noisy: bool = False

    def __repr__(self):  # noqa: D105
        base = "qLogEI" if not self.noisy else "qLogNoisyEI"
        seq = "seq-" if self.sequential else ""

        return f"{seq}{base}"

    def argmax(
        self,
        model,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
        q: int,
        bounds: Float[Tensor, "2 d"],
        options: Optional[dict] = None,
        **kwargs,
    ) -> Float[Tensor, "q d"]:
        """
        Computes the q-Log Expected Improvement.
        """
        options = options or {}

        optimize_device = X.device if self.optimize_device is None else self.optimize_device

        while True:
            if self.noisy:
                qLogEI_acqf = qLogNoisyEI(
                    model=model.to(optimize_device), X_baseline=X.to(optimize_device), prune_baseline=True
                )
            else:
                qLogEI_acqf = qLogEI(model.to(optimize_device), Y.max())

            try:
                bounds_ = bounds.to(optimize_device)
                X_next, values = optimize_acqf(
                    qLogEI_acqf,
                    bounds=bounds_,
                    q=q,
                    gen_candidates=gen_candidates_torch
                    if self.gen_candidates_func == "torch"
                    else gen_candidates_scipy,
                    **options.get("optimize_acqf"),
                    sequential=self.sequential,
                )
                break
            except torch.OutOfMemoryError as e:
                if self.cpu_fallback and self.optimize_device != torch.device("cpu"):
                    model._clear_cache()
                    self.optimize_device = torch.device("cpu")
                else:
                    raise e

        X_next = X_next.detach().to(X)
        return X_next
