"""Score model classes."""

import torch

from ..base import AutonomousModelMixin, GenericModelMixin, Model
from ..utils.nns import ConstantNN

__all__ = [
    "OUScore",
    "DSMScore",
]

class OUScore(AutonomousModelMixin, Model):
    """Analytical score model for Ornstein-Uhlenbeck dynamics.

    Parameters
    ----------
    m0 : torch.Tensor, shape (Ndim,)
        Initial mean.
    S0 : torch.Tensor, shape (Ndim, Ndim)
        Initial covariance.
    D : torch.Tensor, shape (Ndim, Ndim)
        Diffusion matrix.

    Attributes
    ----------
    Ndim_ : int
        Spatial dimension inferred from ``m0`` at construction time.

    """

    def __init__(self, m0, S0, D):
        super(OUScore, self).__init__()
        self.register_buffer("S0", S0)
        self.register_buffer("m0", m0)
        self.register_buffer("D", D)

    def init_model(self, X, generator=None):
        self.Ndim_ = int(X.shape[-1] - 1)
        if self.net_ is None:
            self.net_ = ConstantNN([self.Ndim_, self.Ndim_]).to(
                device=X.device, dtype=X.dtype
            )
        return self

    def dim_output(self):
        """Return output dimension of the constant matrix network.

        Returns
        -------
        int
            Spatial output dimension.
        """
        return self.Ndim_

    def forward(self, Xtrain):
        """Evaluate OU score at ``Xtrain``.

        Parameters
        ----------
        Xtrain : torch.Tensor, shape (batch_size, Ndim + 1)
            Input states with time in the last column.

        Returns
        -------
        torch.Tensor, shape (batch_size, Ndim)
            Score values ``∇_x log p_t(x)``.
        """
        B = self.evaluate_net(Xtrain)
        xt = Xtrain[:, : self.Ndim_]
        t = Xtrain[:, -1]

        t_unique, inv = torch.unique(t, sorted=True, return_inverse=True)

        eBt = torch.matrix_exp((-B)[None, :, :] * t_unique[:, None, None])
        mt = (eBt @ self.m0[:, None]).squeeze(-1)

        eye = torch.eye(self.Ndim_, device=B.device, dtype=B.dtype)
        K = torch.kron(eye, B) + torch.kron(B, eye)
        rhs = (2 * self.D).reshape(-1, 1)
        S_inf = torch.linalg.solve(K, rhs).reshape(self.Ndim_, self.Ndim_)

        eBt_T = torch.transpose(eBt, 1, 2)
        Sigma_t = eBt @ self.S0 @ eBt_T + S_inf - eBt @ S_inf @ eBt_T
        Sigma_inv = torch.linalg.inv(Sigma_t)

        mt_full = mt[inv]
        Sigma_inv_full = Sigma_inv[inv]

        delta = xt - mt_full
        score = -torch.einsum("nij,nj->ni", Sigma_inv_full, delta)
        return score

class DSMScore(GenericModelMixin, Model):
    """Noise-conditional score model with inference-time sigma fallback.

    Parameters
    ----------
    noise_lvl : float, default=0.0
        Default sigma used at inference when sigma is not provided.
    **kwargs : dict
        Forwarded to ``pfi.base.Model``.

    """

    def __init__(
        self,
        noise_lvl=0.0,
        **kwargs,
    ):
        super(DSMScore, self).__init__(**kwargs)
        self.noise_lvl = float(noise_lvl)

    def dim_output(self):
        """Return spatial score output dimension.

        Returns
        -------
        int
            Spatial output dimension.
        """
        return self.Ndim_ - 1

    def _apply_feature_norm(self, x):
        """Apply feature normalization on spatial dimensions only.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor where the final dimensions are ordered as
            ``[..., x_1, ..., x_ndim, sigma, t]``.

        Returns
        -------
        torch.Tensor
            Normalized tensor with sigma/time dimensions left unchanged.
        """
        if not self.feature_norm or "zscore" not in self.feature_norm:
            return x

        x_sp = x[..., : self.Ndim_ - 1]
        center = "nocenter" not in self.feature_norm
        mu = x_sp.mean(dim=-1, keepdim=True) if center else 0
        sd = x_sp.std(dim=-1, keepdim=True, unbiased=True)
        x_sp = (x_sp - mu) / (sd + 1e-5)
        return torch.cat([x_sp, x[..., self.Ndim_ - 1 :]], dim=-1)

    def forward(self, Xtrain):
        """Evaluate DSM score.

        Parameters
        ----------
        Xtrain : torch.Tensor
            Input tensor.
            In training mode, the last dimension must be ``ndim+2`` with
            layout ``[x_1, ..., x_ndim, sigma, t]``.
            In evaluation mode, the last dimension can be either ``ndim+2``
            (explicit sigma) or ``ndim+1`` (sigma injected from
            ``self.noise_lvl``).

        Returns
        -------
        torch.Tensor
            Predicted score on spatial coordinates, shape ``(..., ndim)``.

        Raises
        ------
        ValueError
            If input dimensionality does not match mode requirements.

        Training mode expects ``ndim+2`` features
        ``[x_1, ..., x_ndim, sigma, t]``.
        In evaluation mode, inputs can be either ``ndim+2`` (explicit sigma)
        or ``ndim+1`` (sigma inserted from ``noise_lvl``).
        """
        if self.training:
            if Xtrain.shape[-1] != self.Ndim_ + 1:
                raise ValueError("DSMScore expects ndim+2 features during training.")
            return self.evaluate_net(Xtrain)

        if Xtrain.shape[-1] == self.Ndim_ + 1:
            return self.evaluate_net(Xtrain)

        if Xtrain.shape[-1] == self.Ndim_:
            sigma = torch.full(
                (*Xtrain.shape[:-1], 1),
                self.noise_lvl,
                dtype=Xtrain.dtype,
                device=Xtrain.device,
            )
            Xin = torch.cat([Xtrain[..., :-1], sigma, Xtrain[..., -1:]], dim=-1)
            return self.evaluate_net(Xin)

        raise ValueError("DSMScore expects ndim+1 or ndim+2 features in eval mode.")
