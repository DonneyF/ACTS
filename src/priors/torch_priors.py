"""
Custom prior distributions based on PyTorch.
"""

import math

import torch
from gpytorch.priors import Prior
from gpytorch.priors.utils import _bufferize_attributes
from torch import Tensor
from torch.distributions import AbsTransform, AffineTransform, LogNormal, Normal, TransformedDistribution
from torch.nn import Module as TModule
from torch.types import _size


# Temporary hack to fix https://github.com/cornellius-gp/gpytorch/issues/2581


class LogNormalPrior(Prior, LogNormal):
    """
    Log Normal prior.
    """

    def __init__(self, loc, scale, device: torch.device, validate_args=None, transform=None):
        TModule.__init__(self)
        LogNormal.__init__(self, loc=loc, scale=scale, validate_args=validate_args)
        _bufferize_attributes(self, ("loc", "scale"))
        self._transform = transform
        self.device = device

    def expand(self, batch_shape):  # noqa: D102
        batch_shape = torch.Size(batch_shape)
        return LogNormalPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape), device=self.device)

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x.to(self.device)

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x.to(self.device)


class ShiftedHalfNormalPrior(Prior, TransformedDistribution):
    """
    Half-Normal prior with scaling on x
    pdf(x) = 2 * (2 * pi * scale^2)^-0.5 * exp(-(x-loc)^2 / (2 * scale^2)) for x >= 0; 0 for x < 0
    where scale^2 is the variance.
    """

    def __init__(self, loc, scale, device: torch.device = None, validate_args=None, transform=None):  # noqa: D102
        TModule.__init__(self)

        base_dist = Normal(0, scale, validate_args=False)
        super().__init__(base_dist, [AbsTransform(), AffineTransform(loc=loc, scale=1)], validate_args=validate_args)
        self.loc = torch.tensor(loc, device=device)
        _bufferize_attributes(self, ("loc", "scale"))

        self._transform = transform
        self.device = device

    def expand(self, batch_shape, _instance=None):  # noqa: D102
        batch_shape = torch.Size(batch_shape)
        return ShiftedHalfNormalPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape), device=self.device)

    @property
    def scale(self) -> Tensor:  # noqa: D102
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:  # noqa: D102
        return self.loc + self.scale * math.sqrt(2 / math.pi)

    @property
    def mode(self) -> Tensor:  # noqa: D102
        return self.loc

    @property
    def variance(self) -> Tensor:  # noqa: D102
        return self.scale.pow(2) * (1 - 2 / math.pi)

    def log_prob(self, value):  # noqa: D102
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob = torch.where(value >= 0, log_prob, -math.inf)
        return log_prob
