"""Flow model classes used by flow-matching regression."""

import torch
import torch.nn as nn
from torch.autograd import grad

from ..utils.nns import divergence, _CompoundModel


class CLEFlow(_CompoundModel):
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
    ):
        """Evaluate CLE drift correction terms.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Corrected drift.
        """
        xt = Xtrain[:, 0:self.Ndim].clone()
        xt.requires_grad_(True)
        drift = self.net(xt)
        div_d = divergence(drift, xt)
        with torch.no_grad():
            score = self.score(Xtrain)
        return (
            (drift - self.lx * Xtrain[:, 0:self.Ndim])
            - (0.5 / self.vol) * (div_d + self.lx)
            - (0.5 / self.vol) * (drift + self.lx * Xtrain[:, 0:self.Ndim]) * score
        )


class GradientFlow(_CompoundModel):
    """Gradient flow model parameterized by a scalar potential.

    Parameters
    ----------
    net : torch.nn.Module
        Network mapping ``(batch_size, Ndim)`` to scalar potential values.
    Ndim : int
        State dimension.

    """

    def __init__(
        self,
        net,
        Ndim,
    ):
        super(GradientFlow, self).__init__(net)
        self.Ndim = Ndim

    def forward(
        self,
        Xtrain,
    ):
        """Evaluate negative gradient of potential.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Negative potential gradient.
        """
        xt = Xtrain[:, 0:self.Ndim].clone()
        xt.requires_grad_(True)
        phi = self.net(xt)
        if phi.dim() == 1:
            phi = phi.unsqueeze(-1)
        grad_phi = grad(
            outputs=phi.sum(),
            inputs=xt,
            create_graph=True,
        )[0]
        return -grad_phi


class AutonomousFlow(_CompoundModel):
    """Time-independent drift model.

    Parameters
    ----------
    net : torch.nn.Module
        Drift network acting on state coordinates only.
    Ndim : int
        State dimension.

    """

    def __init__(
        self,
        net,
        Ndim,
    ):
        super(AutonomousFlow, self).__init__(net)
        self.Ndim = Ndim

    def forward(
        self,
        Xtrain,
    ):
        """Evaluate autonomous drift.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            Drift prediction.
        """
        return self.net(Xtrain[:, 0:self.Ndim])


class OUFlow(_CompoundModel):
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
    ):
        """Evaluate OU flow field.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.

        Returns
        -------
        drift : torch.Tensor of shape (batch_size, Ndim)
            OU drift corrected by score and diffusion.
        """
        xt = Xtrain[:, 0:self.Ndim].clone()
        xt.requires_grad_(True)
        with torch.no_grad():
            score = self.score(Xtrain)

        drift = -torch.einsum("mr,nr->nm", self.net, xt)
        return drift - torch.einsum("mr,nr->nm", self.D, score)


class AdditiveFlow(_CompoundModel):
    """Additive-noise flow model with autonomous drift and score correction."""

    def __init__(
        self,
        net,
        score,
        Ndim,
        lx=1.0,
    ):
        super(AdditiveFlow, self).__init__(net)
        self.score = score
        self.Ndim = Ndim
        self.lx = lx

    def forward(
        self,
        Xtrain,
    ):
        xt = Xtrain[:, 0:self.Ndim].clone()
        drift = self.net(xt)
        with torch.no_grad():
            score = self.score(Xtrain)
        return (drift - self.lx * xt) - 0.5 * score


class MultiplicativeFlow(_CompoundModel):
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
    ):
        xt = Xtrain[:, 0:self.Ndim].clone()
        drift = self.net(xt)
        with torch.no_grad():
            score = self.score(Xtrain)
        ones = torch.ones_like(xt)
        return (drift - self.lx * xt) - 0.5 * ones - 0.5 * xt * score
