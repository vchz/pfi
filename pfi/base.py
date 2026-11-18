"""Shared base classes and mixins for flow and score models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .utils.nns import DNN, SpectralNormDNN

class Model(nn.Module, ABC):
    """Abstract model wrapper with optional input normalization.

    Parameters
    ----------
    sizes : list of int or None, default=None
        Hidden-layer widths. If ``None``, defaults to ``[128, 128, 128]``.
    spectral_norm : bool, default=True
        Whether to apply spectral normalization to hidden layers.
    activation : torch.nn.Module or None, default=None
        Hidden activation. If ``None``, defaults to ``ELU``.
    full : bool, default=False
        If ``True``, also apply spectral normalization to the final layer.
    feature_norm : {None, str}, default=None
        Optional feature-wise normalization mode, for example ``"zscore"`` or
        ``"zscore-nocenter"``.
    batch_norm : {None, str}, default=None
        Optional batch-wise normalization mode, for example ``"zscore"`` or
        ``"zscore-notime"``.
    perturb_fn : callable or None, default=None
        Optional post-processing function applied to model outputs.

    Attributes
    ----------
    net_ : torch.nn.Module or None
        Wrapped network instantiated during ``init_model``.
    Ndim_ : int or None
        Inferred spatial dimension set during ``init_model``.

    """

    def __init__(
        self,
        sizes=None,
        spectral_norm=True,
        activation=None,
        full=False,
        feature_norm=None,
        batch_norm=None,
        perturb_fn=None,
    ):
        super(Model, self).__init__()
        self.net_ = None
        self.sizes = sizes
        self.spectral_norm = spectral_norm
        self.activation = activation
        self.full = full
        self.feature_norm = feature_norm
        self.batch_norm = batch_norm
        self.perturb_fn = perturb_fn
        self.register_buffer("scale_mean_", torch.zeros(0))
        self.register_buffer("scale_std_", torch.ones(0))

    @abstractmethod
    def dim_output(self):
        """Return output dimension of the wrapped network.

        Returns
        -------
        int
            Output dimension expected from ``net_``.
        """

    @abstractmethod
    def is_gradient(self):
        """Return whether the model uses a gradient-based output.

        Returns
        -------
        bool
            ``True`` for gradient models, otherwise ``False``.
        """

    @abstractmethod
    def is_autonomous(self):
        """Return whether the model is autonomous in time.

        Returns
        -------
        bool
            ``True`` if the model ignores the time coordinate.
        """

    @abstractmethod
    def _set_dim(self, x):
        """Select network input coordinates from the full state tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose last dimension stores model features.

        Returns
        -------
        torch.Tensor
            Tensor passed to the wrapped network.
        """

    @abstractmethod
    def _set_sign(self, out):
        """Apply model-specific output transform.

        Parameters
        ----------
        out : torch.Tensor
            Raw network output.

        Returns
        -------
        torch.Tensor
            Transformed output, for example after applying ``relu``.
        """

    def dim_spatial(self):
        """Return number of spatial dimension of the wrapped network.

        Returns
        -------
        int
            Number of spatial dimension in net_ (by definition the first dimensions).
        """
        return self.Ndim_
        
    def init_model(self, X, generator=None):
        """Initialize network and normalization statistics from batched data.

        Parameters
        ----------
        X : torch.Tensor
            Batched inputs with last dimension corresponding to features.
            By convention, time is stored in the last coordinate.
        generator : torch.Generator or None, default=None
            Optional RNG for reproducible weight initialization.

        Returns
        -------
        Model
            The initialized model instance.
        """
        self.Ndim_ = int(X.shape[-1] - 1)
        Xn = self._set_dim(X)

        if self.net_ is None:
            sizes = [128, 128, 128] if self.sizes is None else list(self.sizes)
            activation = nn.ELU() if self.activation is None else self.activation
            out_dim = int(self.dim_output())
            full_sizes = [int(Xn.shape[-1])] + sizes + [out_dim]
            if self.spectral_norm:
                self.net_ = SpectralNormDNN(
                    full_sizes,
                    activation=activation,
                    full=self.full,
                    dim_x=self.dim_spatial(),
                    generator=generator,
                )
            else:
                self.net_ = DNN(full_sizes, activation=activation, generator=generator)
            self.net_ = self.net_.to(device=Xn.device, dtype=Xn.dtype)

        Xn = self._apply_feature_norm(Xn)
        dims = tuple(range(Xn.ndim - 1))
        self.scale_mean_ = Xn.mean(dim=dims)
        self.scale_std_ = Xn.std(dim=dims, unbiased=True)
        return self

    def _apply_feature_norm(self, x):
        """Apply optional feature-wise normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor after dimension selection.

        Returns
        -------
        torch.Tensor
            Feature-normalized tensor.
        """
        if not self.feature_norm or "zscore" not in self.feature_norm:
            return x

        dim_spa = int(self.dim_spatial())
        x_sp = x[..., : dim_spa]
        center = "nocenter" not in self.feature_norm
        mu = x_sp.mean(dim=-1, keepdim=True) if center else 0
        sd = x_sp.std(dim=-1, keepdim=True, unbiased=True)
        x_sp = (x_sp - mu) / (sd + 1e-5)
        return torch.cat([x_sp, x[..., dim_spa :]], dim=-1)

    def _apply_batch_norm(self, x):
        """Apply optional batch-wise normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor after feature normalization.

        Returns
        -------
        torch.Tensor
            Batch-normalized tensor.
        """
        if not self.batch_norm or "zscore" not in self.batch_norm:
            return x

        center = "nocenter" not in self.batch_norm
        if center:
            x_out = (x - self.scale_mean_) / (self.scale_std_ + 1e-5)
        else:
            x_out = x / (self.scale_std_ + 1e-5)

        if "notime" in self.batch_norm and x.shape[-1] > self.Ndim_:
            x_out[..., -1] = x[..., -1]
        return x_out

    def _set_scales(self, x):
        """Apply configured feature and batch normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor after dimension selection.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        x = self._apply_feature_norm(x)
        x = self._apply_batch_norm(x)
        return x

    def _set_perturb(self, out):
        """Apply optional perturbation to model outputs.

        Parameters
        ----------
        out : torch.Tensor
            Model output before perturbation.

        Returns
        -------
        torch.Tensor
            Perturbed output if ``perturb_fn`` is defined, otherwise ``out``.
        """
        if self.perturb_fn is None:
            return out
        return self.perturb_fn(out)

    def _check_is_initialized(self):
        """Raise RuntimeError if ``init_model`` has not been called."""
        if self.net_ is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been initialized. "
                "Call init_model() before using the model."
            )

    def evaluate_net(self, x):
        """Evaluate network with dimension selection, scaling, and transform.

        Parameters
        ----------
        x : torch.Tensor
            Full input tensor including spatial coordinates and, by
            convention, time in the last coordinate.

        Returns
        -------
        torch.Tensor
            Final model output after preprocessing and postprocessing.
        """
        self._check_is_initialized()
        x = self._set_dim(x)
        x = self._set_scales(x)
        out = self.net_(x)
        out = self._set_sign(out)
        return self._set_perturb(out)

class AutonomousModelMixin:
    """Mixin for autonomous models using only spatial coordinates.

    """

    def is_autonomous(self):
        """Return ``True`` for autonomous models.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    def is_gradient(self):
        """Return ``False`` for non-gradient models.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    def _set_dim(self, x):
        """Drop time coordinate from network inputs.

        Parameters
        ----------
        x : torch.Tensor
            Full input tensor.

        Returns
        -------
        torch.Tensor
            Spatial coordinates only.
        """
        return x[..., : self.Ndim_]

    def _set_sign(self, out):
        """Leave outputs unchanged.

        Parameters
        ----------
        out : torch.Tensor
            Raw model output.

        Returns
        -------
        torch.Tensor
            Unmodified output.
        """
        return out

class GenericModelMixin:
    """Mixin for non-autonomous models using full input coordinates.

    """

    def is_autonomous(self):
        """Return ``False`` for non-autonomous models.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    def is_gradient(self):
        """Return ``False`` for non-gradient models.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    def _set_dim(self, x):
        """Keep all input coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Full input tensor.

        Returns
        -------
        torch.Tensor
            Unmodified input tensor.
        """
        return x

    def _set_sign(self, out):
        """Leave outputs unchanged.

        Parameters
        ----------
        out : torch.Tensor
            Raw model output.

        Returns
        -------
        torch.Tensor
            Unmodified output.
        """
        return out

class PositiveModelMixin:
    """Mixin enforcing non-negative outputs via ReLU.

    """

    def _set_sign(self, out):
        """Apply ReLU to outputs.

        Parameters
        ----------
        out : torch.Tensor
            Raw model output.

        Returns
        -------
        torch.Tensor
            Non-negative output tensor.
        """
        return torch.relu(out)

class GradientModelMixin:
    """Mixin computing model output as gradient of a scalar potential.

    """

    def evaluate_net(self, x):
        """Evaluate gradient of potential with respect to unscaled inputs.

        Parameters
        ----------
        x : torch.Tensor
            Full input tensor.

        Returns
        -------
        torch.Tensor
            Gradient of the scalar potential with respect to spatial inputs.
        """
        self._check_is_initialized()
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
        """Return ``True`` for gradient-based models.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True
