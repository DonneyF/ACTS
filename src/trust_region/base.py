"""
Base class for trust region methods that, given some parameters, constructs a boundary.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from jaxtyping import Float
from torch import Tensor


@dataclass
class TrustRegion:
    """
    Abstract class for single trust region methods that, given some parameters, constructs a boundary.
    """

    dim: int
    q: int
    n_tot: int
    n_init: int
    state = None
    restart_triggered: bool = False

    def construct_trust_region(
        self, model, X: Float[Tensor, "n d"], Y: Float[Tensor, "n 1"], options: Optional[dict] = None
    ) -> Float[Tensor, "2 d"]:
        """
        Constructs a boundary of trusted region. The returned trust region can exceed [0,1]^d

        :param model: The fitted model
        :param X: Input points
        :param Y: Corresponding targets
        :param options: Optional dictionary of options

        :return trust_region: A 2-by-d tensor of upper and lower bounds
        """
        raise NotImplementedError()

    def update_state(
        self,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
    ) -> None:
        """
        Updates the state based on new observations X, Y

        :param X: Input points
        :param Y: Corresponding targets
        """
        raise NotImplementedError()

    def filter_data(
        self,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
    ) -> Tuple[Float[Tensor, "m d"], Float[Tensor, "m 1"]]:
        """
        Returns the data points contained in the trust region to be used for model training.

        :param X: Input points
        :param Y: Corresponding targets
        :return: Filtered input points and targets
        """
        raise NotImplementedError()

    @property
    def state_dict(self) -> dict:
        """
        :return state_dict: Dictionary corresponding to the state of the trust region
        """
        return {
            key: value for key, value in self.state.__dict__.items() if not key.startswith("__") and not callable(key)
        }
