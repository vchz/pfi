"""Score model classes."""

import torch

from .._models import AutonomousModelMixin, GenericModelMixin, Model
from ..utils.nns import ConstantNN, SpectralNormDNN


class OUScore(AutonomousModelMixin, Model):
    """Analytical score function for Gaussian Ornstein-Uhlenbeck dynamics."""

    def __init__(
        self,
        m0,
        S0,
        D,
        feature_norm=False,
        batch_norm=False,
    ):
        super().__init__(
            sizes=[],
            net_cls=ConstantNN,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        self.Ndim_ = int(m0.shape[0])
        self.register_buffer("S0", S0)
        self.register_buffer("m0", m0)
        self.register_buffer("D", D)

    def dim_output(self):
        return self.Ndim_

    def forward(self, Xtrain):
        B = self.evaluate_net(Xtrain)
        xt = Xtrain[:, 0:self.Ndim_]
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
    """Noise-conditional score wrapper with inference-time sigma fallback."""

    def __init__(
        self,
        sizes=None,
        net_cls=SpectralNormDNN,
        seed=0,
        activation=torch.nn.ELU(),
        noise_lvl=0.0,
        feature_norm=False,
        batch_norm=True,
    ):
        kwargs = dict(
            sizes=sizes,
            net_cls=net_cls,
            seed=seed,
            activation=activation,
            feature_norm=feature_norm,
            batch_norm=batch_norm,
        )
        super().__init__(**kwargs)
        self.noise_lvl = float(noise_lvl)

    def dim_output(self):
        return self.Ndim_ - 1

    def forward(self, Xtrain):
        if self.training:
            if Xtrain.shape[-1] < 2:
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
