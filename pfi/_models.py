"""Shared base classes and mixins for flow and score models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .utils.nns import SpectralNormDNN


class Model(nn.Module, ABC):
    """Abstract model wrapper with optional input normalization.

    Parameters
    ----------
    sizes : list of int or None, default=None
        Hidden-layer widths. If ``None``, defaults to ``[128, 128, 128]``.
    net_cls : type, default=SpectralNormDNN
        Network class instantiated as ``net_cls(full_sizes, **kwargs)`` during
        :meth:`init_model`.
    feature_norm : {None, bool, str}, default=None
        Optional feature-wise normalization mode. ``True`` maps to ``"zscore"``
        and ``False`` maps to ``None``.
    batch_norm : {None, bool, str}, default="zscore"
        Optional batch-wise normalization mode. ``True`` maps to ``"zscore"``
        and ``False`` maps to ``None``.
    perturb_fn : callable or None, default=None
        Optional post-processing function applied to model outputs.
    **kwargs : dict
        Extra keyword arguments forwarded to ``net_cls`` at initialization.
    """

    def __init__(
        self,
        sizes=None,
        net_cls=SpectralNormDNN,
        feature_norm=None,
        batch_norm="zscore",
        perturb_fn=None,
        **kwargs,
    ):
        super(Model, self).__init__()
        self.net_ = None
        self.sizes = [128, 128, 128] if sizes is None else list(sizes)
        self.net_cls = net_cls
        self.net_kwargs = dict(kwargs)

        if feature_norm is True:
            self.feature_norm = "zscore"
        elif feature_norm is False:
            self.feature_norm = None
        else:
            self.feature_norm = feature_norm

        if batch_norm is True:
            self.batch_norm = "zscore"
        elif batch_norm is False:
            self.batch_norm = None
        else:
            self.batch_norm = batch_norm

        self.perturb_fn = perturb_fn
        self.Ndim_ = None
        self.register_buffer("scale_mean", torch.zeros(0))
        self.register_buffer("scale_std", torch.ones(0))

    @abstractmethod
    def dim_output(self):
        """Return output dimension of the wrapped network."""

    @abstractmethod
    def is_gradient(self):
        """Return whether the model uses a gradient-based output."""

    @abstractmethod
    def is_autonomous(self):
        """Return whether the model is autonomous in time."""

    @abstractmethod
    def _set_dim(self, x):
        """Select network input coordinates from the full state tensor."""

    @abstractmethod
    def _set_sign(self, out):
        """Apply model-specific output transform (for example ReLU)."""

    def init_model(self, X):
        """Initialize network and normalization statistics from batched data.

        Parameters
        ----------
        X : torch.Tensor
            Batched inputs with last dimension corresponding to features.
            By convention, time is stored in the last coordinate.

        Returns
        -------
        Model
            The initialized model instance.
        """
        self.Ndim_ = int(X.shape[-1] - 1)
        Xn = self._set_dim(X)

        if self.net_ is None:
            out_dim = int(self.dim_output())
            full_sizes = [int(Xn.shape[-1])] + self.sizes + [out_dim]
            self.net_ = self.net_cls(full_sizes, **self.net_kwargs)
            self.net_ = self.net_.to(device=Xn.device, dtype=Xn.dtype)

        Xn = self._apply_feature_norm(Xn)
        dims = tuple(range(Xn.ndim - 1))
        self.scale_mean = Xn.mean(dim=dims)
        self.scale_std = Xn.std(dim=dims, unbiased=True)
        return self

    def _apply_feature_norm(self, x):
        """Apply optional feature-wise normalization."""
        if not self.feature_norm or "zscore" not in self.feature_norm:
            return x

        x_sp = x[..., : self.Ndim_]
        center = "nocenter" not in self.feature_norm
        mu = x_sp.mean(dim=-1, keepdim=True) if center else 0
        sd = x_sp.std(dim=-1, keepdim=True, unbiased=True)
        x_sp = (x_sp - mu) / (sd + 1e-5)
        return torch.cat([x_sp, x[..., self.Ndim_ :]], dim=-1)

    def _apply_batch_norm(self, x):
        """Apply optional batch-wise normalization."""
        if not self.batch_norm or "zscore" not in self.batch_norm:
            return x
        if self.scale_mean.numel() == 0:
            return x

        center = "nocenter" not in self.batch_norm
        if center:
            x_out = (x - self.scale_mean) / (self.scale_std + 1e-5)
        else:
            x_out = x / (self.scale_std + 1e-5)

        if "notime" in self.batch_norm and x.shape[-1] > self.Ndim_:
            x_out[..., -1] = x[..., -1]
        return x_out

    def _set_scales(self, x):
        """Apply configured feature and batch normalization."""
        x = self._apply_feature_norm(x)
        x = self._apply_batch_norm(x)
        return x

    def _set_perturb(self, out):
        """Apply optional perturbation to model outputs."""
        if self.perturb_fn is None:
            return out
        return self.perturb_fn(out)

    def evaluate_net(self, x):
        """Evaluate network with dimension selection, scaling, and transform."""
        x = self._set_dim(x)
        x = self._set_scales(x)
        out = self.net_(x)
        out = self._set_sign(out)
        return self._set_perturb(out)


class AutonomousModelMixin:
    """Mixin for autonomous models using only spatial coordinates."""

    def is_autonomous(self):
        """Return ``True`` for autonomous models."""
        return True

    def is_gradient(self):
        """Return ``False`` for non-gradient models."""
        return False

    def _set_dim(self, x):
        """Drop time coordinate from network inputs."""
        return x[..., : self.Ndim_]

    def _set_sign(self, out):
        """Leave outputs unchanged."""
        return out


class GenericModelMixin:
    """Mixin for non-autonomous models using full input coordinates."""

    def is_autonomous(self):
        """Return ``False`` for non-autonomous models."""
        return False

    def is_gradient(self):
        """Return ``False`` for non-gradient models."""
        return False

    def _set_dim(self, x):
        """Keep all input coordinates."""
        return x

    def _set_sign(self, out):
        """Leave outputs unchanged."""
        return out


class PositiveModelMixin:
    """Mixin enforcing non-negative outputs via ReLU."""

    def _set_sign(self, out):
        """Apply ReLU to outputs."""
        return torch.relu(out)


class GradientModelMixin:
    """Mixin computing model output as gradient of a scalar potential."""

    def evaluate_net(self, x):
        """Evaluate gradient of potential with respect to unscaled inputs."""
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
        grad = self._set_sign(grad)
        return self._set_perturb(grad)

    def is_gradient(self):
        """Return ``True`` for gradient-based models."""
        return True
