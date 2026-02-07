"""Flow model classes used by flow-matching regression."""

import torch
import torch.nn as nn

from ..utils.nns import divergence


class CLEFlow(nn.Module):
    """Chemical-Langevin-inspired flow model.

    Parameters
    ----------
    drift_model : torch.nn.Module
        Drift network taking ``(batch_size, Ndim)`` inputs.
    score : torch.nn.Module
        Score model taking ``(batch_size, Ndim + 1)`` inputs.
    Ndim : int
        State dimension.
    vol : float, default=1.0
        Volume scaling of stochastic corrections.
    lx : float, default=1.0
        Linear degradation coefficient.

    Attributes
    ----------
    drift_model : torch.nn.Module
        Drift network.
    score : torch.nn.Module
        Score network.
    Ndim : int
        State dimension.
    vol : float
        Volume scale.
    lx : float
        Linear degradation coefficient.
    """

    def __init__(
        self,
        drift_model,
        score,
        Ndim,
        vol=1.0,
        lx=1.0,
    ):
        super(CLEFlow, self).__init__()
        self.drift_model = drift_model
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
        drift = self.drift_model(xt)
        div_d = divergence(drift, xt)
        with torch.no_grad():
            score = self.score(Xtrain)
        return (
            (drift - self.lx * Xtrain[:, 0:self.Ndim])
            - (0.5 / self.vol) * (div_d + self.lx)
            - (0.5 / self.vol) * (drift + self.lx * Xtrain[:, 0:self.Ndim]) * score
        )


class GradientFlow(nn.Module):
    """Gradient flow model parameterized by a scalar potential.

    Parameters
    ----------
    potential_model : torch.nn.Module
        Network mapping ``(batch_size, Ndim)`` to scalar potential values.
    Ndim : int
        State dimension.

    Attributes
    ----------
    potential_model : torch.nn.Module
        Potential network.
    Ndim : int
        State dimension.
    """

    def __init__(
        self,
        potential_model,
        Ndim,
    ):
        super(GradientFlow, self).__init__()
        self.potential_model = potential_model
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
        phi = self.potential_model(xt)
        if phi.dim() == 1:
            phi = phi.unsqueeze(-1)
        grad_phi = grad(
            outputs=phi.sum(),
            inputs=xt,
            create_graph=True,
        )[0]
        return -grad_phi


class AutonomousFlow(nn.Module):
    """Time-independent drift model.

    Parameters
    ----------
    drift_model : torch.nn.Module
        Drift network acting on state coordinates only.
    Ndim : int
        State dimension.

    Attributes
    ----------
    drift_model : torch.nn.Module
        Drift network.
    Ndim : int
        State dimension.
    """

    def __init__(
        self,
        drift_model,
        Ndim,
    ):
        super(AutonomousFlow, self).__init__()
        self.drift_model = drift_model
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
        return self.drift_model(Xtrain[:, 0:self.Ndim])


class OUFlow(nn.Module):
    """Ornstein-Uhlenbeck flow model using an external score function.

    Parameters
    ----------
    B : torch.Tensor of shape (ndim, ndim)
        Drift matrix.
    score : torch.nn.Module
        Score model mapping ``(batch_size, ndim + 1)`` to ``(batch_size, ndim)``.
    D : torch.Tensor of shape (ndim, ndim)
        Diffusion matrix.

    Attributes
    ----------
    score : torch.nn.Module
        Score model.
    Ndim : int
        State dimension.
    B : torch.nn.Parameter of shape (ndim, ndim)
        Learnable drift matrix.
    D : torch.Tensor of shape (ndim, ndim)
        Diffusion matrix.
    """

    def __init__(
        self,
        B,
        score,
        D,
    ):
        super(OUFlow, self).__init__()
        self.score = score
        self.Ndim = D.shape[0]
        self.B = nn.Parameter(B)
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

        drift = -torch.einsum("mr,nr->nm", self.B, xt)
        return drift - torch.einsum("mr,nr->nm", self.D, score)
