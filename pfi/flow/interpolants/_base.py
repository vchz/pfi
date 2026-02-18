"""Base interpolant interface used by flow-matching solvers."""

from abc import ABC, abstractmethod


class BaseInterpolant(ABC):
    """Abstract base class for trajectory interpolants."""

    def __init__(self, device="cpu"):
        self.device = device

    def fit(self, nodes_fit, Dist):
        """Store fit data and call implementation-specific fitting."""
        self.y_fit_ = Dist
        self.t_fit_ = nodes_fit
        self._fit()
        return self

    @abstractmethod
    def predict(self, t_eval):
        """Evaluate interpolant and derivative at requested nodes."""
        return self._predict(t_eval)

    @abstractmethod
    def _fit(self):
        """Fit internal interpolant state."""
        pass
