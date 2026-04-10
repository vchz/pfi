"""Flow model classes used by flow-matching regression."""

import torch
import torch.nn as nn

from .._models import (
    Model,
    AutonomousModelMixin,
    GenericModelMixin,
    PositiveModelMixin,
    GradientModelMixin,
)
from ..utils.nns import SpectralNormDNN, divergence, symsqrt
from ..utils.nns import LinearNN


class CLEFlow(AutonomousModelMixin, Model):
    """Chemical-Langevin-inspired autonomous flow model."""

    def __init__(
        self,
        score,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=nn.ELU(),
        vol=1.0,
        lx=1.0,
        feature_norm=False,
        batch_norm=True,
    ):
        """Initialize CLE flow components."""
        super(CLEFlow, self).__init__(
            sizes=sizes,
            net_cls=net_cls,
            seed=seed,
            activation=activation,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        self.score = score
        self.vol = vol
        self.lx = lx

    def dim_output(self):
        return self.Ndim_

    def forward(
        self,
        Xtrain,
        stoch=False,
    ):
        """Evaluate CLE drift correction terms.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return the stochastic drift part
            ``drift - lx * x``. If ``False``, return the probability flow
            drift

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Corrected drift (when ``stoch=False``).
        tuple : (drift, noise)
            When ``stoch=True``, returns drift and sampled noise term.
        """
        with torch.enable_grad():
            Xeval = Xtrain.clone().requires_grad_(True)
            xt = Xeval[:, : self.Ndim_]
            drift = self.evaluate_net(Xeval)
            DD = drift + self.lx * xt
            drift_part = drift - self.lx * xt

            if stoch:
                noise = torch.sqrt(torch.relu(DD))* torch.randn_like(xt)
                return drift_part, noise
            
            div_d = divergence(DD, xt, create_graph=self.training)

        with torch.no_grad():
            score = self.score(Xtrain)
        
        return (
            drift_part - (0.5 / self.vol)*(div_d + DD*score)
        )
    

class PositiveCLEFlow(PositiveModelMixin, CLEFlow):
    """Positive-output CLE flow using ``ReLU`` on the drift-network output.

    This class reuses :class:`CLEFlow` dynamics and overrides
    ``evaluate_net`` through :class:`PositiveModelMixin`.
    """

    pass


class AutonomousGrowth(AutonomousModelMixin, Model):
    """Autonomous growth model wrapper over a neural network."""

    def __init__(
        self,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=nn.ELU(),
        feature_norm=False,
        batch_norm=True,
    ):
        super(AutonomousGrowth, self).__init__(
            sizes=sizes,
            net_cls=net_cls,
            seed=seed,
            activation=activation,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )

    def dim_output(self):
        return 1

    def forward(self, Xtrain):
        return self.evaluate_net(Xtrain)


class OUFlow(AutonomousModelMixin, Model):
    """Ornstein-Uhlenbeck flow model with linear drift matrix."""

    def __init__(
        self,
        score,
        D=0.5,
        lx=0,
        feature_norm=False,
        batch_norm=False,
    ):
        """Initialize OU flow components."""
        super(OUFlow, self).__init__(
            sizes=[],
            net_cls=LinearNN,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        self.lx = lx
        self.score = score
        self.register_buffer("D", torch.as_tensor(D))

    def dim_output(self):
        return self.Ndim_

    def forward(
        self,
        Xtrain,
        stoch=False,
    ):
        """Evaluate OU flow field.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return the stochastic drift part ``-B x``.
            If ``False``, return the probability flow drift
            ``-B x - D score``.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            OU drift corrected by score and diffusion (when ``stoch=False``).
        tuple : (drift, noise)
            When ``stoch=True``, returns drift and sampled noise term.
        """
        xt = Xtrain[:, : self.Ndim_].clone()
        xt.requires_grad_(True)
        drift = - self.evaluate_net(Xtrain) - self.lx * xt
        D = self.D.to(device=xt.device, dtype=xt.dtype)
        if D.dim() == 0:
            D = D * torch.eye(self.Ndim_, dtype=xt.dtype, device=xt.device)

        if stoch:
            noise = torch.einsum("mr,nr->nm", D, torch.randn_like(xt))
            return drift, noise
        with torch.no_grad():
            score = self.score(Xtrain)

        return drift - torch.einsum("mr,nr->nm", D, score)


class AutonomousODEFlow(AutonomousModelMixin, Model):
    """Autonomous deterministic flow with linear degradation."""

    def __init__(
        self,
        score,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=nn.ELU(),
        lx=1.0,
        feature_norm=False,
        batch_norm=True,
    ):
        """Initialize autonomous ODE flow components."""
        super(AutonomousODEFlow, self).__init__(
            sizes=sizes,
            net_cls=net_cls,
            seed=seed,
            activation=activation,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        self.score = score
        self.lx = lx

    def dim_output(self):
        return self.Ndim_

    def forward(
        self,
        Xtrain,
        stoch=False,
    ):
        """Evaluate autonomous drift with linear degradation.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return the stochastic drift part ``drift - lx*x``.
            If ``False``, return ``drift - lx*x``.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Deterministic drift field (when ``stoch=False``).
        tuple : (drift, noise)
            When ``stoch=True``, returns drift and a zero-noise placeholder.
        """
        xt = Xtrain[:, : self.Ndim_].clone()
        drift = self.evaluate_net(Xtrain)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = 0
            return drift_part, noise
        
            
        return (drift - self.lx * xt)


class AdditiveGradientFlow(GradientModelMixin, AutonomousModelMixin, Model):
    """Autonomous additive-noise flow with gradient-based drift."""

    def __init__(
        self,
        score,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=nn.ELU(),
        lx=1.0,
        D=0.5,
        feature_norm=False,
        batch_norm=True,
    ):
        """Initialize additive gradient flow components."""
        super(AdditiveGradientFlow, self).__init__(
            sizes=sizes,
            net_cls=net_cls,
            seed=seed,
            activation=activation,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        self.score = score
        self.lx = lx
        self.register_buffer("D", torch.as_tensor(D))

    def dim_output(self):
        return 1

    def forward(
        self,
        Xtrain,
        stoch=False,
    ):
        """Evaluate additive gradient-model drift.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return the stochastic drift part ``drift - lx*x``.
            If ``False``, return the probability flow drift with score term.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Additive gradient-model drift field (when ``stoch=False``).
        tuple : (drift, noise)
            When ``stoch=True``, returns drift and sampled additive noise term.
        """
        with torch.enable_grad():
            Xeval = Xtrain.clone().requires_grad_(True)
            drift = self.evaluate_net(Xeval)
            xt = Xeval[:, : self.Ndim_]
        D = self.D.to(device=xt.device, dtype=xt.dtype)
        if D.dim() == 0:
            D = D * torch.eye(self.Ndim_, dtype=xt.dtype, device=xt.device)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = torch.einsum("mr,nr->nm", symsqrt(2.0 * D), torch.randn_like(xt))
            return drift_part, noise
        with torch.no_grad():
            score = self.score(Xtrain)
        return (drift - self.lx * xt) - torch.einsum("mr,nr->nm", D, score)


class NonAutonomousAdditiveGradientFlow(GradientModelMixin, GenericModelMixin, Model):
    """Non-autonomous additive-noise flow with gradient-based drift."""

    def __init__(
        self,
        score,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=nn.ELU(),
        lx=1.0,
        D=0.5,
        feature_norm=False,
        batch_norm=True,
    ):
        """Initialize non-autonomous additive gradient flow components."""
        super(NonAutonomousAdditiveGradientFlow, self).__init__(
            sizes=sizes,
            net_cls=net_cls,
            seed=seed,
            activation=activation,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        self.score = score
        self.lx = lx
        self.register_buffer("D", torch.as_tensor(D))

    def dim_output(self):
        return 1

    def forward(
        self,
        Xtrain,
        stoch=False,
    ):
        """Evaluate non-autonomous additive gradient-model drift.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return the stochastic drift part ``drift - lx*x``.
            If ``False``, return the probability flow drift with score term.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Additive gradient-model drift field (when ``stoch=False``).
        tuple : (drift, noise)
            When ``stoch=True``, returns drift and sampled additive noise term.
        """
        with torch.enable_grad():
            xt_full = Xtrain.clone()
            xt_full.requires_grad_(True)
            drift = self.evaluate_net(xt_full)[..., : self.Ndim_]
            xt = xt_full[:, : self.Ndim_]
        D = self.D.to(device=xt.device, dtype=xt.dtype)
        if D.dim() == 0:
            D = D * torch.eye(self.Ndim_, dtype=xt.dtype, device=xt.device)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = torch.einsum("mr,nr->nm", symsqrt(2.0 * D), torch.randn_like(xt))
            return drift_part, noise
        with torch.no_grad():
            score = self.score(Xtrain)
        return (drift - self.lx * xt) - torch.einsum("mr,nr->nm", D, score)

    
class ODEFlow(GenericModelMixin, Model):
    """Deterministic non-autonomous flow with linear degradation."""

    def __init__(
        self,
        score,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=nn.ELU(),
        lx=1.0,
        feature_norm=False,
        batch_norm=True,
    ):
        """Initialize ODE flow components."""
        super(ODEFlow, self).__init__(
            sizes=sizes,
            net_cls=net_cls,
            seed=seed,
            activation=activation,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        self.score = score
        self.lx = lx

    def dim_output(self):
        return self.Ndim_

    def forward(
        self,
        Xtrain,
        stoch=False,
    ):
        """Evaluate time-conditioned drift with linear degradation.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return the stochastic drift part ``drift - lx*x``.
            If ``False``, return ``drift - lx*x``.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Deterministic drift field (when ``stoch=False``).
        tuple : (drift, noise)
            When ``stoch=True``, returns drift and a zero-noise placeholder.
        """
        xt = Xtrain[:, :].clone()
        drift = self.evaluate_net(Xtrain)

        if stoch:
            drift_part = drift - self.lx * xt[:, : self.Ndim_ ]
            noise = 0
            return drift_part, noise
        
        return (drift - self.lx * xt[:, : self.Ndim_])
    
    
class AdditiveFlow(AutonomousModelMixin, Model):
    """Autonomous additive-noise flow with score correction."""

    def __init__(
        self,
        score,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=nn.ELU(),
        lx=1.0,
        D=0.5,
        feature_norm=False,
        batch_norm=True,
    ):
        """Initialize additive flow components."""
        super(AdditiveFlow, self).__init__(
            sizes=sizes,
            net_cls=net_cls,
            seed=seed,
            activation=activation,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        self.score = score
        self.lx = lx
        self.register_buffer("D", torch.as_tensor(D))

    def dim_output(self):
        return self.Ndim_

    def forward(
        self,
        Xtrain,
        stoch=False,
    ):
        """Evaluate additive model drift.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return the stochastic drift part ``drift - lx*x``.
            If ``False``, return the probability flow drift with score term.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Additive-model drift field (when ``stoch=False``).
        tuple : (drift, noise)
            When ``stoch=True``, returns drift and sampled additive noise term.
        """
        xt = Xtrain[:, : self.Ndim_].clone()
        D = self.D.to(device=xt.device, dtype=xt.dtype)
        if D.dim() == 0:
            D = D * torch.eye(self.Ndim_, dtype=xt.dtype, device=xt.device)
        drift = self.evaluate_net(Xtrain)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = torch.einsum("mr,nr->nm", symsqrt(2.0 * D), torch.randn_like(xt))
            return drift_part, noise
        
        with torch.no_grad():
            score = self.score(Xtrain)
            
        return (drift - self.lx * xt) - torch.einsum("mr,nr->nm", D, score)


class MultiplicativeFlow(AutonomousModelMixin, Model):
    """Multiplicative-noise flow model with autonomous drift and score correction."""

    def __init__(
        self,
        score,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=nn.ELU(),
        lx=1.0,
        feature_norm=False,
        batch_norm=True,
    ):
        """Initialize multiplicative flow components."""
        super(MultiplicativeFlow, self).__init__(
            sizes=sizes,
            net_cls=net_cls,
            seed=seed,
            activation=activation,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        self.score = score
        self.lx = lx

    def dim_output(self):
        return self.Ndim_

    def forward(
        self,
        Xtrain,
        stoch=False,
    ):
        """Evaluate multiplicative model drift.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return the stochastic drift part ``drift - lx*x``.
            If ``False``, return the probability flow drift with score term.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Multiplicative-model drift field (when ``stoch=False``).
        tuple : (drift, noise)
            When ``stoch=True``, returns drift and sampled multiplicative
            noise term.
        """
        xt = Xtrain[:, : self.Ndim_].clone()
        drift = self.evaluate_net(Xtrain)
        drift_part = drift - self.lx * xt

        if stoch:
            noise = torch.sqrt(xt) * torch.randn_like(xt)
            return drift_part, noise
        
        with torch.no_grad():
            score = self.score(Xtrain)

        ones = torch.ones_like(xt)
        return drift_part - 0.5 * ones - 0.5 * xt * score
