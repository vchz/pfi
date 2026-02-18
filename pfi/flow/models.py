"""Flow model classes used by flow-matching regression."""

import torch
import torch.nn as nn
from ..utils.nns import divergence, symsqrt




class CLEFlow(nn.Module):
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

    def set_scales(self, mean, std):
        """Set input normalization statistics on the drift network.

        Parameters
        ----------
        mean : torch.Tensor
            Feature-wise input mean.
        std : torch.Tensor
            Feature-wise input standard deviation.

        Returns
        -------
        self : CLEFlow
            Estimator instance.
        """
        if hasattr(self.net, "set_scales"):
            self.net.set_scales(mean, std)
        return self

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
            drift = self.net(xt)
            if stoch:
                drift_part = drift - self.lx * Xtrain[:, 0:self.Ndim]
                noise = torch.sqrt(torch.relu(drift + self.lx * Xtrain[:, 0:self.Ndim])) * torch.randn_like(xt)
                return drift_part, noise
            div_d = divergence(drift, xt, create_graph=self.training)
        with torch.no_grad():
            score = self.score(Xtrain)
        return (
            (drift - self.lx * Xtrain[:, 0:self.Ndim])
            - (0.5 / self.vol) * (div_d + self.lx)
            - (0.5 / self.vol) * (drift + self.lx * Xtrain[:, 0:self.Ndim]) * score
        )
    

class OUFlow(nn.Module):
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
        self.D = D

    def set_scales(self, mean, std):
        """Set input normalization statistics on the drift network.

        Parameters
        ----------
        mean : torch.Tensor
            Feature-wise input mean.
        std : torch.Tensor
            Feature-wise input standard deviation.

        Returns
        -------
        self : OUFlow
            Estimator instance.
        """
        if hasattr(self.net, "set_scales"):
            self.net.set_scales(mean, std)
        return self

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

class AutonomousODEFlow(nn.Module):
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
    D : float or torch.Tensor, default=1.0
        Kept for API consistency with other flow classes.
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

    def set_scales(self, mean, std):
        """Set input normalization statistics on the drift network.

        Parameters
        ----------
        mean : torch.Tensor
            Feature-wise input mean.
        std : torch.Tensor
            Feature-wise input standard deviation.

        Returns
        -------
        self : AutonomousODEFlow
            Estimator instance.
        """
        if hasattr(self.net, "set_scales"):
            self.net.set_scales(mean, std)
        return self

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
        drift = self.net(xt)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = 0
            return drift_part, noise
        
            
        return (drift - self.lx * xt)


class ODEFlow(nn.Module):
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
    D : float or torch.Tensor, default=1.0
        Kept for API consistency with other flow classes.
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

    def set_scales(self, mean, std):
        """Set input normalization statistics on the drift network.

        Parameters
        ----------
        mean : torch.Tensor
            Feature-wise input mean.
        std : torch.Tensor
            Feature-wise input standard deviation.

        Returns
        -------
        self : ODEFlow
            Estimator instance.
        """
        if hasattr(self.net, "set_scales"):
            self.net.set_scales(mean, std)
        return self

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
        xt = Xtrain[:, :].clone()
        drift = self.net(Xtrain)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = 0
            return drift_part, noise
        
        return (drift - self.lx * xt)
    
    
class AdditiveFlow(nn.Module):
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
    D : float or torch.Tensor, default=1.0
        Diffusion coefficient/matrix used for score correction and stochastic
        sampling.
    """

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
        D=1.0,
    ):
        """Initialize additive flow components."""
        super(AdditiveFlow, self).__init__()
        self.net = net
        self.score = score
        self.Ndim = Ndim
        self.lx = lx
        self.D = D

    def set_scales(self, mean, std):
        """Set input normalization statistics on the drift network.

        Parameters
        ----------
        mean : torch.Tensor
            Feature-wise input mean.
        std : torch.Tensor
            Feature-wise input standard deviation.

        Returns
        -------
        self : AdditiveFlow
            Estimator instance.
        """
        if hasattr(self.net, "set_scales"):
            self.net.set_scales(mean, std)
        return self

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
        drift = self.net(xt)

        if stoch:
            drift_part = drift - self.lx * xt
            noise = torch.einsum("mr,nr->nm", symsqrt(2.0*self.D), torch.randn_like(xt))
            return drift_part, noise
        
        with torch.no_grad():
            score = self.score(Xtrain)
            
        return (drift - self.lx * xt) - torch.einsum(
            "mr,nr->nm", self.D, score
        )


class MultiplicativeFlow(nn.Module):
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

    def set_scales(self, mean, std):
        """Set input normalization statistics on the drift network.

        Parameters
        ----------
        mean : torch.Tensor
            Feature-wise input mean.
        std : torch.Tensor
            Feature-wise input standard deviation.

        Returns
        -------
        self : MultiplicativeFlow
            Estimator instance.
        """
        if hasattr(self.net, "set_scales"):
            self.net.set_scales(mean, std)
        return self

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
        drift = self.net(xt)
        if stoch:
            drift_part = drift - self.lx * xt
            noise = torch.sqrt(torch.relu(xt)) * torch.randn_like(xt)
            return drift_part, noise
        
        with torch.no_grad():
            score = self.score(Xtrain)

        ones = torch.ones_like(xt)
        return (drift - self.lx * xt) - 0.5 * ones - 0.5 * xt * score


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
        self.S0 = S0
        self.m0 = m0
        self.D = D

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
