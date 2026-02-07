"""Public flow-regression APIs, models, and solver modules.

This submodule provides flow regression estimators and the building blocks
needed to train and evaluate flow models.
"""

from ._base import FlowRegression
from . import models, solvers, interpolants

__all__ = [
    "FlowRegression",
    "models",
    "solvers",
    "interpolants",
]
