"""Score models used by score-matching."""

import torch
import torch.nn as nn


class OUScore(nn.Module):
    """Analytical score function for Gaussian Ornstein-Uhlenbeck dynamics.

    Parameters
    ----------
    B : torch.Tensor of shape (ndim, ndim)
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
        B,
        m0,
        S0,
        D,
    ):
        super(OUScore, self).__init__()
        self.Ndim = m0.shape[0]
        self.S0 = S0
        self.B = nn.Parameter(B)
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

        eBt = torch.matrix_exp((-self.B)[None, :, :] * t_unique[:, None, None])
        mt = (eBt @ self.m0[:, None]).squeeze(-1)

        eye = torch.eye(self.Ndim, device=self.B.device, dtype=self.B.dtype)
        K = torch.kron(eye, self.B) + torch.kron(self.B, eye)
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
