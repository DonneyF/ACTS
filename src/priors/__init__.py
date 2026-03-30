"""
Module for prior distributions.
"""

from .torch_priors import LogNormalPrior, ShiftedHalfNormalPrior


__all__ = [
    "LogNormalPrior",
    "ShiftedHalfNormalPrior",
]
