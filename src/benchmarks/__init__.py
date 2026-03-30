# ruff: noqa: F401
"""
Benchmark tasks.
"""

from importlib import import_module
from typing import TYPE_CHECKING


# Public names -> relative module path + attribute name
_LAZY_ATTRS: dict[str, str] = {
    "LassoDna": ".lassobench_task:LassoDna",
    "Mopta08": ".mopta08:Mopta08",
    "Rover": ".rover:Rover",
    "SVM": ".svm:SVM",
    "Ant": ".mujoco:Ant",
    "Cheetah": ".mujoco:Cheetah",
    "Hopper": ".mujoco:Hopper",
    "Humanoid": ".mujoco:Humanoid",
    "Swimmer": ".mujoco:Swimmer",
    "Walker": ".mujoco:Walker",
}


def __getattr__(name: str):
    """
    Lazy-load tasks so `from src.benchmarks import Ant` (and Hydra _target_)
    doesn't import every heavy dependency at package import time.
    """
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_name, attr_name = target.split(":", 1)
    mod = import_module(mod_name, package=__name__)
    obj = getattr(mod, attr_name)

    # Cache so subsequent lookups are fast and avoid re-importing
    globals()[name] = obj
    return obj


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))


def guacamol_benchmark_factory(**kwargs):
    """
    Factory function to create a Guacamol benchmark based on the specified task ID.
    """
    from src.benchmarks.guacamol_selfies_vae import GuacamolObjective
    return GuacamolObjective(**kwargs)


__all__ = [*list(_LAZY_ATTRS.keys())]


# For type checkers / IDEs (does not execute at runtime)
if TYPE_CHECKING:
    from .lassobench_task import LassoDna
    from .mopta08 import Mopta08
    from .mujoco import Ant, Cheetah, Hopper, Humanoid, Swimmer, Walker
    from .rover import Rover
    from .svm import SVM
    from .guacamol_selfies_vae import GuacamolObjective