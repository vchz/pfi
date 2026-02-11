"""Flow model classes used by flow-matching regression."""

import torch
import torch.nn as nn
from torch.autograd import grad

from ..utils.nns import divergence, CompoundModel, symsqrt




class CLEFlow(CompoundModel):
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
        super(CLEFlow, self).__init__(net)
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
    

class OUFlow(CompoundModel):
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
        super(OUFlow, self).__init__(nn.Parameter(net))
        self.score = score
        self.Ndim = D.shape[0]
        self.B = self.net
        self.D = D

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

class AutonomousODEFlow(CompoundModel):
    """Istropic aditive-noise flow model with autonomous drift and score correction."""

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
        D=1.0,
    ):
        super(AutonomousODEFlow, self).__init__(net)
        self.score = score
        self.Ndim = Ndim
        self.lx = lx
        self.D = D

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


class ODEFlow(CompoundModel):
    """Istropic aditive-noise flow model with autonomous drift and score correction."""

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
        D=1.0,
    ):
        super(ODEFlow, self).__init__(net)
        self.score = score
        self.Ndim = Ndim
        self.lx = lx
        self.D = D

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
    
    
class AdditiveFlow(CompoundModel):
    """Istropic aditive-noise flow model with autonomous drift and score correction."""

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
        D=1.0,
    ):
        super(AdditiveFlow, self).__init__(net)
        self.score = score
        self.Ndim = Ndim
        self.lx = lx
        self.D = D

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


class MultiplicativeFlow(CompoundModel):
    """Multiplicative-noise flow model with autonomous drift and score correction."""

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
    ):
        super(MultiplicativeFlow, self).__init__(net)
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
        drift = self.net(xt)
        if stoch:
            drift_part = drift - self.lx * xt
            noise = torch.sqrt(torch.relu(xt)) * torch.randn_like(xt)
            return drift_part, noise
        
        with torch.no_grad():
            score = self.score(Xtrain)

        ones = torch.ones_like(xt)
        return (drift - self.lx * xt) - 0.5 * ones - 0.5 * xt * score
