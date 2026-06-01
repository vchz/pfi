"""Flow model classes used by flow-matching solvers."""

import torch

from .._models import (
    AutonomousModelMixin,
    GenericModelMixin,
    GradientModelMixin,
    Model,
    PositiveModelMixin,
)
from ..utils.nns import LinearNN, divergence, symsqrt

__all__ = [
    "CLEFlow",
    "CLELambdaFlow",
    "PositiveCLEFlow",
    "AutonomousGrowth",
    "OUFlow",
    "AutonomousODEFlow",
    "AdditiveGradientFlow",
    "NonAutonomousAdditiveGradientFlow",
    "ODEFlow",
    "AdditiveFlow",
    "MultiplicativeFlow",
]


class CLEFlow(AutonomousModelMixin, Model):
    """Chemical-Langevin-inspired autonomous flow model."""

    def __init__(
        self,
        score,
        vol=1.0,
        lx=1.0,
        **kwargs,
    ):
        """Initialize CLE flow model.

        Parameters
        ----------
        score : callable
            Score model called on ``Xtrain``.
        vol : float, default=1.0
            Volume scale used in the CLE correction term.
        lx : float, default=1.0
            Linear degradation coefficient.
        **kwargs : dict
            Forwarded to :class:`pfi._models.Model`.
        """
        super(CLEFlow, self).__init__(**kwargs)
        self.score = score
        self.vol = vol
        self.lx = lx

    def dim_output(self):
        """Return network output dimension."""
        return self.Ndim_

    def forward(self, Xtrain, stoch=False):
        """Evaluate drift or stochastic CLE terms.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return drift and noise. Otherwise return probability
            flow drift.

        Returns
        -------
        torch.Tensor or tuple
            Deterministic corrected drift when ``stoch=False``;
            ``(drift_part, noise)`` when ``stoch=True``.
        """
        with torch.enable_grad():
            Xeval = Xtrain.clone().requires_grad_(True)
            xt = Xeval[:, : self.Ndim_]
            drift = self.evaluate_net(Xeval)
            DD = drift + self.lx * xt
            drift_part = drift - self.lx * xt

            if stoch:
                noise = torch.sqrt(torch.relu(DD)) * torch.randn_like(xt)
                return drift_part, noise

            div_d = divergence(DD, xt, create_graph=self.training)

        with torch.no_grad():
            score = self.score(Xtrain)

        return drift_part - (0.5 / self.vol) * (div_d + DD * score)


class CLELambdaFlow(AutonomousModelMixin, Model):
    """CLE flow model with trainable degradation vector."""

    def __init__(
        self,
        score,
        vol=1.0,
        lx=1.0,
        **kwargs,
    ):
        """Initialize CLE flow model with trainable degradation.

        Parameters
        ----------
        score : callable
            Score model called on ``Xtrain``.
        vol : float, default=1.0
            Volume scale used in the CLE correction term.
        lx : float, default=1.0
            Initial value used to initialize ``lambda_``.
        **kwargs : dict
            Forwarded to :class:`pfi._models.Model`.
        """
        super(CLELambdaFlow, self).__init__(**kwargs)
        self.score = score
        self.vol = vol
        self.lx = lx
        self.lambda_ = None

    def init_model(self, X):
        """Initialize network, scales, and trainable ``lambda_``.

        Parameters
        ----------
        X : torch.Tensor
            Batched input tensor used to initialize the parent model.

        Returns
        -------
        CLELambdaFlow
            The initialized model instance.
        """
        super(CLELambdaFlow, self).init_model(X)
        self.lambda_ = torch.nn.Parameter(self.lx * torch.ones(1, device=X.device, dtype=X.dtype))
        return self

    def dim_output(self):
        """Return network output dimension."""
        return self.Ndim_

    def forward(self, Xtrain, stoch=False):
        """Evaluate CLE drift with learned degradation correction.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return drift and sampled stochastic noise.
            If ``False``, return probability flow drift.

        Returns
        -------
        torch.Tensor or tuple
            Deterministic corrected drift when ``stoch=False``;
            ``(drift_part, noise)`` when ``stoch=True``.
        """
        with torch.enable_grad():
            Xeval = Xtrain.clone().requires_grad_(True)
            xt = Xeval[:, : self.Ndim_]
            drift = self.evaluate_net(Xeval)
            lambda_vec = torch.relu(self.lambda_).to(device=xt.device, dtype=xt.dtype)
            DD = drift + lambda_vec * xt
            drift_part = drift - lambda_vec * xt

            if stoch:
                noise = torch.sqrt(torch.relu(DD)) * torch.randn_like(xt)
                return drift_part, noise

            div_d = divergence(DD, xt, create_graph=self.training)

        with torch.no_grad():
            score = self.score(Xtrain)

        return drift_part - (0.5 / self.vol) * (div_d + DD * score)


class PositiveCLEFlow(PositiveModelMixin, CLEFlow):
    """Positive-output CLE flow (ReLU applied to network output)."""

    pass


class AutonomousGrowth(AutonomousModelMixin, Model):
    """Autonomous growth model wrapping a scalar neural network output."""

    def __init__(self, **kwargs):
        """Initialize autonomous growth model.

        Parameters
        ----------
        **kwargs : dict
            Forwarded to :class:`pfi._models.Model`.
        """
        super(AutonomousGrowth, self).__init__(**kwargs)

    def dim_output(self):
        """Return scalar growth output dimension."""
        return 1

    def forward(self, Xtrain):
        """Evaluate autonomous growth field.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.

        Returns
        -------
        torch.Tensor, shape (batch_size, 1)
            Predicted growth values.
        """
        return self.evaluate_net(Xtrain)


class OUFlow(AutonomousModelMixin, Model):
    """Ornstein-Uhlenbeck flow model with linear drift matrix."""

    def __init__(
        self,
        score,
        D=0.5,
        lx=0,
        **kwargs,
    ):
        """Initialize OU flow model.

        Parameters
        ----------
        score : callable
            Score model called on ``Xtrain``.
        D : float or torch.Tensor, default=0.5
            Diffusion matrix/scalar.
        lx : float, default=0
            Linear degradation coefficient.
        **kwargs : dict
            Forwarded to :class:`pfi._models.Model` after overriding
            ``net_cls`` with :class:`pfi.utils.nns.LinearNN`.
        """
        kwargs.pop("sizes", None)
        kwargs.pop("net_cls", None)
        kwargs.pop("feature_norm", None)
        kwargs.pop("batch_norm", None)
        super(OUFlow, self).__init__(
            sizes=[],
            net_cls=LinearNN,
            feature_norm=False,
            batch_norm=False,
            **kwargs,
        )
        self.lx = lx
        self.score = score
        self.register_buffer("D", torch.as_tensor(D))

    def dim_output(self):
        """Return network output dimension."""
        return self.Ndim_

    def forward(self, Xtrain, stoch=False):
        """Evaluate OU drift or stochastic terms.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return drift and sampled stochastic noise.
            If ``False``, return probability flow drift.

        Returns
        -------
        torch.Tensor or tuple
            Deterministic corrected drift when ``stoch=False``;
            ``(drift, noise)`` when ``stoch=True``.
        """
        xt = Xtrain[:, : self.Ndim_].clone()
        xt.requires_grad_(True)
        drift = -self.evaluate_net(xt) - self.lx * xt

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
        lx=1.0,
        **kwargs,
    ):
        """Initialize autonomous ODE flow model."""
        super(AutonomousODEFlow, self).__init__(**kwargs)
        self.score = score
        self.lx = lx

    def dim_output(self):
        """Return network output dimension."""
        return self.Ndim_

    def forward(self, Xtrain, stoch=False):
        """Evaluate autonomous deterministic drift.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return drift and a zero-noise placeholder.
            If ``False``, return deterministic drift.

        Returns
        -------
        torch.Tensor or tuple
            Deterministic drift when ``stoch=False``;
            ``(drift_part, 0)`` when ``stoch=True``.
        """
        xt = Xtrain[:, : self.Ndim_].clone()
        drift = self.evaluate_net(Xtrain)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = 0
            return drift_part, noise

        return drift - self.lx * xt


class AdditiveGradientFlow(GradientModelMixin, AutonomousModelMixin, Model):
    """Autonomous additive-noise flow with gradient drift."""

    def __init__(
        self,
        score,
        lx=1.0,
        D=0.5,
        **kwargs,
    ):
        """Initialize additive gradient flow model."""
        super(AdditiveGradientFlow, self).__init__(**kwargs)
        self.score = score
        self.lx = lx
        self.register_buffer("D", torch.as_tensor(D))

    def dim_output(self):
        """Return scalar potential output dimension."""
        return 1

    def forward(self, Xtrain, stoch=False):
        """Evaluate additive gradient drift or stochastic terms.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return drift and sampled additive noise.
            If ``False``, return probability flow drift.

        Returns
        -------
        torch.Tensor or tuple
            Deterministic corrected drift when ``stoch=False``;
            ``(drift_part, noise)`` when ``stoch=True``.
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
    """Non-autonomous additive-noise flow with gradient drift."""

    def __init__(
        self,
        score,
        lx=1.0,
        D=0.5,
        **kwargs,
    ):
        """Initialize non-autonomous additive gradient flow model."""
        super(NonAutonomousAdditiveGradientFlow, self).__init__(**kwargs)
        self.score = score
        self.lx = lx
        self.register_buffer("D", torch.as_tensor(D))

    def dim_output(self):
        """Return scalar potential output dimension."""
        return 1

    def forward(self, Xtrain, stoch=False):
        """Evaluate non-autonomous additive gradient drift.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return drift and sampled additive noise.
            If ``False``, return probability flow drift.

        Returns
        -------
        torch.Tensor or tuple
            Deterministic corrected drift when ``stoch=False``;
            ``(drift_part, noise)`` when ``stoch=True``.
        """
        with torch.enable_grad():
            xt_full = Xtrain.clone().requires_grad_(True)
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
        lx=1.0,
        **kwargs,
    ):
        """Initialize non-autonomous ODE flow model."""
        super(ODEFlow, self).__init__(**kwargs)
        self.score = score
        self.lx = lx

    def dim_output(self):
        """Return network output dimension."""
        return self.Ndim_

    def forward(self, Xtrain, stoch=False):
        """Evaluate non-autonomous deterministic drift.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return drift and a zero-noise placeholder.
            If ``False``, return deterministic drift.

        Returns
        -------
        torch.Tensor or tuple
            Deterministic drift when ``stoch=False``;
            ``(drift_part, 0)`` when ``stoch=True``.
        """
        xt = Xtrain[:, : self.Ndim_].clone()
        drift = self.evaluate_net(Xtrain)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = 0
            return drift_part, noise

        return drift - self.lx * xt


class AdditiveFlow(AutonomousModelMixin, Model):
    """Autonomous additive-noise flow with score correction."""

    def __init__(
        self,
        score,
        lx=1.0,
        D=0.5,
        **kwargs,
    ):
        """Initialize additive flow model."""
        super(AdditiveFlow, self).__init__(**kwargs)
        self.score = score
        self.lx = lx
        self.register_buffer("D", torch.as_tensor(D))

    def dim_output(self):
        """Return network output dimension."""
        return self.Ndim_

    def forward(self, Xtrain, stoch=False):
        """Evaluate additive drift or stochastic terms.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return drift and sampled additive noise.
            If ``False``, return probability flow drift.

        Returns
        -------
        torch.Tensor or tuple
            Deterministic corrected drift when ``stoch=False``;
            ``(drift_part, noise)`` when ``stoch=True``.
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
    """Multiplicative-noise flow model with score correction."""

    def __init__(
        self,
        score,
        lx=1.0,
        **kwargs,
    ):
        """Initialize multiplicative flow model."""
        super(MultiplicativeFlow, self).__init__(**kwargs)
        self.score = score
        self.lx = lx

    def dim_output(self):
        """Return network output dimension."""
        return self.Ndim_

    def forward(self, Xtrain, stoch=False):
        """Evaluate multiplicative drift or stochastic terms.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, return drift and sampled multiplicative noise.
            If ``False``, return probability flow drift.

        Returns
        -------
        torch.Tensor or tuple
            Deterministic corrected drift when ``stoch=False``;
            ``(drift_part, noise)`` when ``stoch=True``.
        """
        xt = Xtrain[:, : self.Ndim_].clone()
        drift = self.evaluate_net(Xtrain)
        drift_part = drift - self.lx * xt

        if stoch:
            noise = torch.sqrt(xt) * torch.randn_like(xt)
            return drift_part, noise

        with torch.no_grad():
            score = self.score(Xtrain)

        return drift_part - 0.5 * torch.ones_like(xt) - 0.5 * xt * score
