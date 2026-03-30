"""
TurBO trust region
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from src.trust_region.base import TrustRegion


@dataclass
class Turbo(TrustRegion):
    """
    Turbo trust regions

    Eriksson, David, et al. Scalable global optimization via local Bayesian optimization.
    Advances in Neural Information Processing Systems. 2019
    """

    geometry: Literal["rectangular", "cylindrical"] = "rectangular"

    length: float = float("nan")  # Note: Post-initialized
    length_min: float = float("nan")  # Note: Post-initialized
    length_max: float = float("nan")  # Note: Post-initialized

    # CTS Sigma
    sigma: float = 0.125
    sigma_min: float = 0.5**7
    sigma_max: float = 0.125  # Same as initial sigma value.

    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized

    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")

    def __post_init__(self):
        """Initialize TurBO parameters based on geometry."""
        # Standard (Rectangular) TurBO
        if self.geometry == "rectangular":
            self.length: float = 0.8
            self.length_min: float = 0.5**7
            self.length_max: float = 1.6

            self.failure_tolerance = math.ceil(max([4.0 / self.q, float(self.dim) / self.q]))
        else:
            # Cylindrical TurBO
            rho_init: float = 1.0
            rho_min: float = 0.01

            self.length = rho_init * np.sqrt(self.dim)
            self.length_min = rho_min * np.sqrt(self.dim)
            self.length_max = self.length

            n_fails_to_min = np.ceil(-np.log2(self.length_min / self.length))  # fails to reach R_min
            budget_after_init = self.n_tot - self.n_init
            original_failtol = np.ceil(np.max([4.0 / self.q, self.dim / self.q]))
            custom_failtol = np.ceil(0.5 * budget_after_init / (self.q * n_fails_to_min))
            self.failure_tolerance = np.min([original_failtol, custom_failtol])

    def update_state(self, X, Y):
        """Update the TurBO state. Sets restart_triggered"""
        if max(Y) > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.sigma = min([2.0 * self.sigma, self.sigma_max])
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.sigma /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, max(Y).item())
        if self.length < self.length_min:
            # Note: CTS does not check sigma_min for restart
            self.restart_triggered = True
        return self

    def construct_trust_region(
        self, model, X: Float[Tensor, "n d"], Y: Float[Tensor, "n 1"], options: Optional[dict] = None
    ) -> Float[Tensor, "2 d"]:
        """Construct rectangular trust region around the incumbent."""
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / weights.numel()))
        tr_lb = x_center - weights * self.length / 2.0
        tr_ub = x_center + weights * self.length / 2.0
        bounds = torch.stack([tr_lb, tr_ub])

        return bounds

    def filter_data(
        self,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
    ) -> Tuple[Float[Tensor, "n d"], Float[Tensor, "n 1"]]:
        """Filters the training data, but TurBO trains on all data"""
        return X, Y
