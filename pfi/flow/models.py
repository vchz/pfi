"""Flow model classes used by flow-matching regression."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from ..utils.nns import divergence, symsqrt


class Flow(nn.Module, ABC):
    """Abstract base class for flow models."""

    @abstractmethod
    def set_scales(self, mean, std):
        """Set normalization statistics on the underlying network."""

    @abstractmethod
    def evaluate_net(self, x):
        """Evaluate the underlying drift network."""


class AutonomousFlowMixin:
    """Scale handling for autonomous models (spatial dimensions only)."""

    def set_scales(self, mean, std):
        if hasattr(self.net, "set_scales"):
            self.net.set_scales(mean[: self.Ndim], std[: self.Ndim])
        return self


class NonAutonomousFlowMixin:
    """Scale handling for non-autonomous models (all input dimensions)."""

    def set_scales(self, mean, std):
        if hasattr(self.net, "set_scales"):
            self.net.set_scales(mean, std)
        return self


class PositiveFlowMixin:
    """Positive drift evaluation mixin."""

    def evaluate_net(self, x):
        return torch.relu(self.net(x))


class GenericFlowMixin:
    """Unconstrained drift evaluation mixin."""

    def evaluate_net(self, x):
        return self.net(x)


class CLEFlow(AutonomousFlowMixin, GenericFlowMixin, Flow):
    """Chemical-Langevin-inspired flow model.

    Parameters
    ----------
    net : torch.nn.Module
        Drift network taking ``(batch_size, Ndim)`` inputs.
    score : torch.nn.Module
        Score model taking ``(batch_size, Ndim + 1)`` inputs.
    Ndim : int
        State dimension.
    vol : float, default=1.0
        Volume scaling of stochastic corrections.
    lx : float, default=1.0
        Linear degradation coefficient.

    """

    def __init__(
        self,
        net,
        score,
        Ndim,
        vol=1.0,
        lx=1.0,
    ):
        """Initialize CLE flow components."""
        super(CLEFlow, self).__init__()
        self.net = net
        self.score = score
        self.Ndim = Ndim
        self.vol = vol
        self.lx = lx

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
            xt = Xtrain[:, 0:self.Ndim].clone()
            xt.requires_grad_(True)
            drift = self.evaluate_net(xt)
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
    

class PositiveCLEFlow(PositiveFlowMixin, CLEFlow):
    """Positive-output CLE flow using ``ReLU`` on the drift-network output.

    This class reuses :class:`CLEFlow` dynamics and overrides
    ``evaluate_net`` through :class:`PositiveFlowMixin`.
    """

    pass


class OUFlow(AutonomousFlowMixin, GenericFlowMixin, Flow):
    """Ornstein-Uhlenbeck flow model using an external score function.

    Parameters
    ----------
    net : torch.Tensor of shape (ndim, ndim)
        Drift matrix.
    score : torch.nn.Module
        Score model mapping ``(batch_size, ndim + 1)`` to ``(batch_size, ndim)``.
    D : torch.Tensor of shape (ndim, ndim)
        Diffusion matrix.

    """

    def __init__(
        self,
        net,
        score,
        D,
    ):
        """Initialize OU flow components."""
        super(OUFlow, self).__init__()
        self.net = nn.Parameter(net)
        self.score = score
        self.Ndim = D.shape[0]
        self.B = self.net
        self.register_buffer("D", D)

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
        xt = Xtrain[:, 0:self.Ndim].clone()
        xt.requires_grad_(True)
        drift = -torch.einsum("mr,nr->nm", self.net, xt)
        if stoch:
            noise = torch.einsum("mr,nr->nm", self.D, torch.randn_like(xt))
            return drift, noise
        with torch.no_grad():
            score = self.score(Xtrain)

        return drift - torch.einsum("mr,nr->nm", self.D, score)


class AutonomousODEFlow(AutonomousFlowMixin, GenericFlowMixin, Flow):
    """Autonomous deterministic flow with linear degradation.

    Parameters
    ----------
    net : torch.nn.Module
        Drift network taking ``(batch_size, Ndim)`` inputs.
    score : torch.nn.Module
        Kept for API consistency with other flow classes.
    Ndim : int
        State dimension.
    lx : float, default=1.0
        Linear degradation coefficient.
    """

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
    ):
        """Initialize autonomous ODE flow components."""
        super(AutonomousODEFlow, self).__init__()
        self.net = net
        self.score = score
        self.Ndim = Ndim
        self.lx = lx

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
        xt = Xtrain[:, 0:self.Ndim].clone()
        drift = self.evaluate_net(xt)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = 0
            return drift_part, noise
        
            
        return (drift - self.lx * xt)


class AdditiveGradientFlow(AutonomousFlowMixin, GenericFlowMixin, Flow):
    """Additive-noise flow with gradient-based drift.

    The drift is defined as the gradient of a scalar potential network and
    combined with additive-noise probability-flow correction, as in
    ``AdditiveFlow``.

    Parameters
    ----------
    net : torch.nn.Module
        Scalar potential network taking ``(batch_size, Ndim)`` inputs and
        returning either shape ``(batch_size,)`` or ``(batch_size, 1)``.
    score : torch.nn.Module
        Score model mapping ``(batch_size, Ndim + 1)`` to ``(batch_size, Ndim)``.
    Ndim : int
        State dimension.
    lx : float, default=1.0
        Linear degradation coefficient.
    D : float or torch.Tensor, default=0.5
        Diffusion coefficient/matrix used for score correction and stochastic
        sampling.
    """

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
        D=0.5,
    ):
        """Initialize additive gradient flow components."""
        super(AdditiveGradientFlow, self).__init__()
        self.net = net
        self.score = score
        self.Ndim = Ndim
        self.lx = lx
        ref_param = next(self.net.parameters())
        dtype = ref_param.dtype
        device = ref_param.device
        D = torch.as_tensor(D, device=device, dtype=dtype)
        if D.dim() == 0:
            D = D * torch.eye(self.Ndim, dtype=dtype, device=device)
        self.register_buffer("D", D)

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
            xt = Xtrain[:, 0:self.Ndim].clone()
            xt.requires_grad_(True)
            potential = self.evaluate_net(xt).squeeze(-1)
            drift = torch.autograd.grad(
                potential.sum(),
                xt,
                create_graph=self.training,
            )[0]

        if stoch:
            drift_part = drift - self.lx * xt
            noise = torch.einsum("mr,nr->nm", symsqrt(2.0 * self.D), torch.randn_like(xt))
            return drift_part, noise
        with torch.no_grad():
            score = self.score(Xtrain)
        return (drift - self.lx * xt) - torch.einsum("mr,nr->nm", self.D, score)


class NonAutonomousAdditiveGradientFlow(NonAutonomousFlowMixin, GenericFlowMixin, Flow):
    """Non-autonomous additive-noise flow with gradient-based drift.

    The potential network is evaluated on full time-augmented inputs
    ``(x, t)``, but only the spatial gradient ``d/dx`` is used as drift.

    Parameters
    ----------
    net : torch.nn.Module
        Scalar potential network taking ``(batch_size, Ndim + 1)`` inputs and
        returning either shape ``(batch_size,)`` or ``(batch_size, 1)``.
    score : torch.nn.Module
        Score model mapping ``(batch_size, Ndim + 1)`` to ``(batch_size, Ndim)``.
    Ndim : int
        State dimension.
    lx : float, default=1.0
        Linear degradation coefficient.
    D : float or torch.Tensor, default=0.5
        Diffusion coefficient/matrix used for score correction and stochastic
        sampling.
    """

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
        D=0.5,
    ):
        """Initialize non-autonomous additive gradient flow components."""
        super(NonAutonomousAdditiveGradientFlow, self).__init__()
        self.net = net
        self.score = score
        self.Ndim = Ndim
        self.lx = lx
        ref_param = next(self.net.parameters())
        dtype = ref_param.dtype
        device = ref_param.device
        D = torch.as_tensor(D, device=device, dtype=dtype)
        if D.dim() == 0:
            D = D * torch.eye(self.Ndim, dtype=dtype, device=device)
        self.register_buffer("D", D)

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
            potential = self.evaluate_net(xt_full).squeeze(-1)
            grad_full = torch.autograd.grad(
                potential.sum(),
                xt_full,
                create_graph=self.training,
            )[0]
            drift = grad_full[:, : self.Ndim]
            xt = xt_full[:, : self.Ndim]

        if stoch:
            drift_part = drift - self.lx * xt
            noise = torch.einsum("mr,nr->nm", symsqrt(2.0 * self.D), torch.randn_like(xt))
            return drift_part, noise
        with torch.no_grad():
            score = self.score(Xtrain)
        return (drift - self.lx * xt) - torch.einsum("mr,nr->nm", self.D, score)

    
class ODEFlow(NonAutonomousFlowMixin, GenericFlowMixin, Flow):
    """Deterministic flow with drift network on full time-augmented inputs.

    Parameters
    ----------
    net : torch.nn.Module
        Drift network taking ``(batch_size, Ndim + 1)`` inputs.
    score : torch.nn.Module
        Kept for API consistency with other flow classes.
    Ndim : int
        State dimension.
    lx : float, default=1.0
        Linear degradation coefficient.
    """

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
    ):
        """Initialize ODE flow components."""
        super(ODEFlow, self).__init__()
        self.net = net
        self.score = score
        self.Ndim = Ndim
        self.lx = lx

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
            drift_part = drift - self.lx * xt[:,:self.Ndim]
            noise = 0
            return drift_part, noise
        
        return (drift - self.lx * xt[:,:self.Ndim])
    
    
class AdditiveFlow(AutonomousFlowMixin, GenericFlowMixin, Flow):
    """Additive-noise flow with score-based probability-flow correction.

    Parameters
    ----------
    net : torch.nn.Module
        Drift network taking ``(batch_size, Ndim)`` inputs.
    score : torch.nn.Module
        Score model mapping ``(batch_size, Ndim + 1)`` to ``(batch_size, Ndim)``.
    Ndim : int
        State dimension.
    lx : float, default=1.0
        Linear degradation coefficient.
    D : float or torch.Tensor, default=0.5
        Diffusion coefficient/matrix used for score correction and stochastic
        sampling.
    """

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
        D=0.5,
    ):
        """Initialize additive flow components."""
        super(AdditiveFlow, self).__init__()
        self.net = net
        self.score = score
        self.Ndim = Ndim
        self.lx = lx
        ref_param = next(self.net.parameters())
        dtype = ref_param.dtype
        device = ref_param.device
        D = torch.as_tensor(D, device=device, dtype=dtype)
        if D.dim() == 0:
            D = D * torch.eye(self.Ndim, dtype=dtype, device=device)
        self.register_buffer("D", D)

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
        xt = Xtrain[:, 0:self.Ndim].clone()
        drift = self.evaluate_net(xt)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = torch.einsum("mr,nr->nm", symsqrt(2.0 * self.D), torch.randn_like(xt))
            return drift_part, noise
        
        with torch.no_grad():
            score = self.score(Xtrain)
            
        return (drift - self.lx * xt) - torch.einsum("mr,nr->nm", self.D, score)


class MultiplicativeFlow(AutonomousFlowMixin, GenericFlowMixin, Flow):
    """Multiplicative-noise flow model with autonomous drift and score correction."""

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
    ):
        """Initialize multiplicative flow components."""
        super(MultiplicativeFlow, self).__init__()
        self.net = net
        self.score = score
        self.Ndim = Ndim
        self.lx = lx

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
        xt = Xtrain[:, 0:self.Ndim].clone()
        drift = self.evaluate_net(xt)
        drift_part = drift - self.lx * xt

        if stoch:
            noise = torch.sqrt(xt) * torch.randn_like(xt)
            return drift_part, noise
        
        with torch.no_grad():
            score = self.score(Xtrain)

        ones = torch.ones_like(xt)
        return drift_part - 0.5 * ones - 0.5 * xt * score


class OUScore(nn.Module):
    """Analytical score function for Gaussian Ornstein-Uhlenbeck dynamics.

    Parameters
    ----------
    net : torch.Tensor of shape (ndim, ndim)
        Drift matrix.
    m0 : torch.Tensor of shape (ndim,)
        Initial mean.
    S0 : torch.Tensor of shape (ndim, ndim)
        Initial covariance matrix.
    D : torch.Tensor of shape (ndim, ndim)
        Diffusion matrix.

    """

    def __init__(
        self,
        net,
        m0,
        S0,
        D,
    ):
        """Initialize analytical OU score parameters."""
        super(OUScore, self).__init__()
        self.net = nn.Parameter(net)
        self.Ndim = m0.shape[0]
        self.register_buffer("S0", S0)
        self.register_buffer("m0", m0)
        self.register_buffer("D", D)

    def forward(
        self,
        Xtrain,
    ):
        """Evaluate score values at time-stamped states.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, ndim + 1)
            Input where the last column is time.

        Returns
        -------
        score : torch.Tensor of shape (batch_size, ndim)
            Analytical score values.
        """
        xt = Xtrain[:, 0:self.Ndim]
        t = Xtrain[:, -1]

        t_unique, inv = torch.unique(t, sorted=True, return_inverse=True)

        eBt = torch.matrix_exp((-self.net)[None, :, :] * t_unique[:, None, None])
        mt = (eBt @ self.m0[:, None]).squeeze(-1)

        eye = torch.eye(self.Ndim, device=self.net.device, dtype=self.net.dtype)
        K = torch.kron(eye, self.net) + torch.kron(self.net, eye)
        rhs = (2 * self.D).reshape(-1, 1)
        S_inf = torch.linalg.solve(K, rhs).reshape(self.Ndim, self.Ndim)

        eBt_T = torch.transpose(eBt, 1, 2)
        Sigma_t = eBt @ self.S0 @ eBt_T + S_inf - eBt @ S_inf @ eBt_T
        Sigma_inv = torch.linalg.inv(Sigma_t)

        mt_full = mt[inv]
        Sigma_inv_full = Sigma_inv[inv]

        delta = xt - mt_full
        score = -torch.einsum("nij,nj->ni", Sigma_inv_full, delta)

        return score
