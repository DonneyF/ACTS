"""
Spherical trust region based on intersection of a sphere centered at [-1,1]^d and one centered at the incumbnent with
radius length.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from src.trust_region.base import TrustRegion


@dataclass
class SphereCap(TrustRegion):
    """
    A spherical trust region that filters data based on Euclidean distance from the incumbent on the sphere.

    Specifically, captures points $x$ such that $||x - x_0|| <= length$
    Equivalently, $x^\top x_0 >= r^2 - length^2 / 2$ where $||x|| = ||x_0|| = r$
    """

    length: float = 0.1

    length_min: float = float("nan")  # Note: Post-initialized
    length_max: float = float("nan")  # Note: Post-initialized

    best_y: float = -float("inf")

    def construct_trust_region(
        self, model, X: Float[Tensor, "n d"], Y: Float[Tensor, "n 1"], options: Optional[dict] = None
    ) -> Float[Tensor, "2 d"]:
        """Constructs rectangular bounding box around the incumbent."""
        x_incumbent = X[Y.argmax()]

        lb = x_incumbent - self.length
        ub = x_incumbent + self.length
        return torch.stack([lb, ub])

    def update_state(
        self,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
    ) -> None:
        """Update the SphereCap state. Sets best_value."""
        self.best_y = max(self.best_y, max(Y).item())

    def filter_data(
        self,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
    ) -> Tuple[Float[Tensor, "m d"], Float[Tensor, "m 1"]]:
        """
        Keeps only points x such that ||x - x_0|| <= epsilon, where x_0 is the incumbent

        X is assumed to be in [0,1]^d.
        """
        if len(Y) == 0:
            return X, Y

        X_tf = X * 2 - 1  # Convert to [-1,1]^d
        x0 = X_tf[Y.argmax()]
        mask = torch.norm(X_tf - x0, dim=-1) <= self.length

        return X[mask], Y[mask]

    def project(
        self,
        x: Float[Tensor, "n d"],
        x_incumbent: Float[Tensor, "1 d"] | Float[Tensor, "d"],  # noqa: F821
    ) -> Float[Tensor, "n d"]:
        """Projects x onto the sphere cap trust region, centered at x_incumbent. Both points are on the unit sphere."""
        # assert torch.allclose(torch.norm(x_incumbent, dim=-1), torch.tensor(1.0).to(x_incumbent))
        # assert torch.allclose(torch.norm(x, dim=-1), torch.ones_like(x[:, 0]))

        if len(x_incumbent.shape) == 1:
            x_incumbent = x_incumbent.unsqueeze(0)

        c = 1 - self.length**2 / 2
        u = x - x @ x_incumbent.T * x_incumbent
        u = u / torch.norm(u, dim=-1, keepdim=True)

        # Project: X_next = c * x_incumbent + sqrt(1 - c^2) * u
        # We add some small weight to the direction of the incumbent due to avoid being exactly on the boundary.
        X_project = (c + 1e-6) * x_incumbent + math.sqrt(1 - c**2) * u

        # For points where ||x - x_incumbent||_2 <= eps, do nothing.
        in_tr = (x @ x_incumbent.T >= c).squeeze(-1)
        X_project[in_tr] = x[in_tr]

        return X_project
