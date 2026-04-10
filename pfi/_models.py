"""Shared model base classes and mixins for flow/score modules."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from .utils.nns import SpectralNormDNN


class Model(nn.Module, ABC):
    """Abstract base class with optional feature/batch normalization."""

    def __init__(
        self,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=nn.ELU(),
        feature_norm=False,
        batch_norm=True,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.net_ = None
        self.sizes = [128, 128, 128] if sizes is None else list(sizes)
        self.net_cls = net_cls
        self.seed = seed
        self.activation = activation
        self.feature_norm = feature_norm
        self.batch_norm = batch_norm
        self.norm_eps = norm_eps
        self.Ndim_ = None
        self.register_buffer("scale_mean", torch.zeros(0))
        self.register_buffer("scale_std", torch.ones(0))

    @abstractmethod
    def dim_output(self):
        """Return output dimension of the model network."""

    def init_model(self, X):
        """Initialize normalization statistics from batched inputs ``X``."""
        self.Ndim_ = int(X.shape[-1] - 1)
        Xn = self._set_dim(X)
        if self.net_ is None:
            out_dim = int(self.dim_output())
            full_sizes = [int(Xn.shape[-1])] + self.sizes + [out_dim]
            self.net_ = self.net_cls(full_sizes, seed=self.seed, activation=self.activation)
            self.net_ = self.net_.to(device=Xn.device, dtype=Xn.dtype)
        Xn = self._apply_feature_norm(Xn)
        dims = tuple(range(Xn.ndim - 1))
        self.scale_mean = Xn.mean(dim=dims)
        self.scale_std = Xn.std(dim=dims, unbiased=True)
        return self

    @abstractmethod
    def is_gradient(self):
        """Return whether the model is a gradient."""
    
    @abstractmethod
    def is_autonomous(self):
        """Return whether the model is autonomous."""

    @abstractmethod
    def _set_dim(self, x):
        """Select network input coordinates from the full model state."""

    @abstractmethod
    def _set_sign(self, out):
        """Apply optional output transform."""

    def _apply_feature_norm(self, x):
        if not self.feature_norm:
            return x
        mu = x[...,:self.Ndim_].mean(dim=-1, keepdim=True)
        sd = x[...,:self.Ndim_].std(dim=-1, keepdim=True, unbiased=True)
        return (x - 0*mu) / (sd + self.norm_eps)

    def _apply_batch_norm(self, x):
        if not self.batch_norm:
            return x
        if self.scale_mean.numel() == 0:
            return x
        return (x - self.scale_mean) / (self.scale_std + self.norm_eps)

    def _set_scales(self, x):
        x = self._apply_feature_norm(x)
        x = self._apply_batch_norm(x)
        return x

    def evaluate_net(self, x):
        """Evaluate the underlying network with model-specific transforms."""
        x = self._set_dim(x)
        x = self._set_scales(x)
        return self._set_sign(self.net_(x))


class AutonomousModelMixin:
    """Input-dimension handling for autonomous models."""

    def is_autonomous(self):
        return True

    def is_gradient(self):
        return False
    
    def _set_dim(self, x):
        return x[..., : self.Ndim_]

    def _set_sign(self, out):
        return out


class GenericModelMixin:
    """Input-dimension handling for generic non-autonomous models."""

    def is_autonomous(self):
        return False

    def is_gradient(self):
        return False
    
    def _set_dim(self, x):
        return x

    def _set_sign(self, out):
        return out


class PositiveModelMixin:
    """Positive-output evaluation mixin."""

    def _set_sign(self, out):
        return torch.relu(out)


class GradientModelMixin:
    """Gradient-based evaluation mixin."""

    def evaluate_net(self, x):
        x_dim = self._set_dim(x)
        if not x_dim.requires_grad:
            x_dim = x_dim.detach().clone().requires_grad_(True)
        x_scaled = self._set_scales(x_dim)
        potential = self.net_(x_scaled).squeeze(-1)
        grad = torch.autograd.grad(
            potential.sum(),
            x_dim,
            create_graph=self.training,
        )[0]

        grad = grad[..., : self.Ndim_]
        return self._set_sign(grad)

    def is_gradient(self):
        return True
