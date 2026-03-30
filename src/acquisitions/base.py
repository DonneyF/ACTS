"""
Base class for generic Acquisition
"""

from typing import Optional

from jaxtyping import Float
from torch import Tensor


class Acquisition:
    """
    Base class for generic Acquisition
    """

    # Flag to indicate if the acquisition proposes points independent of the surrogate model
    random: bool = False

    # Flag to indicate if the acquisition proposes points that are unbounded
    unbounded: bool = False

    # Flag to indicate if the acquisition function optimizes over a fixed radius of the search space
    radius_constrained: bool = False

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
        Optimize an acquisition function.

        :param model: Fitted model.
        :param X: Tensor of input points
        :param Y: Tensor of corresponding targets, not standardized (same scale as true objective)
        :param q: Batch size
        :param bounds: d-dimensional Tensor of upper and lower bounds to optimize acquisition over
        :param options: Optional dictionary of options
        :param kwargs: Additional keyword arguments

        :return X: The q points that maximizes the acquisition function
        """
        raise NotImplementedError()
