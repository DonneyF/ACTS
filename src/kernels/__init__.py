"""
Factories for creating covariance modules for BoTorch models.
"""

from typing import Optional

from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, RBFKernel


def matern_factory(d: int, *, dsp: bool, **kwargs) -> MaternKernel:
    """
    Factory function to create a Matern 5/2 kernel with the specified number of dimensions.

    :param d: The number of dimensions for the kernel.
    :param dsp: If True, use a dimension-scaled prior for the lengthscale.
    """
    if dsp:
        return get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel=False, **kwargs)
    else:
        return MaternKernel(ard_num_dims=d, nu=2.5, **kwargs)


def rbf_factory(d: int, *, dsp: bool, **kwargs) -> RBFKernel:
    """
    Factory function to create an RBF kernel with the specified number of dimensions.

    :param d: The number of dimensions for the kernel.
    :param dsp: If True, use a dimension-scaled prior for the lengthscale.
    """
    if dsp:
        return get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel=True, **kwargs)
    else:
        return RBFKernel(ard_num_dims=d, **kwargs)


__all__ = [
    "matern_factory",
    "rbf_factory",
]
