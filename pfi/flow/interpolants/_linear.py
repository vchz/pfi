"""Piecewise-linear interpolant for trajectory data."""

import torch

from ._base import BaseInterpolant


class LinearInterpolant(BaseInterpolant):
    """Batched piecewise-linear interpolant.

    Parameters
    ----------
    device : str or torch.device, default='cpu'
        Computation device inherited from `pfi.flow.interpolants.BaseInterpolant`.

    Attributes
    ----------
    y_fit_ : torch.Tensor of shape (batch_size, n_nodes, ndim)
        Training trajectories, set by `fit`.
    t_fit_ : torch.Tensor of shape (batch_size, n_nodes)
        Training time nodes, set by `fit`.
    """

    def _fit(self):
        """No-op fit for linear interpolation.

        Returns
        -------
        self : LinearInterpolant
            Fitted interpolant.
        """
        return self

    def predict(
        self,
        t_eval,
    ):
        """Evaluate linear interpolation and piecewise-constant derivative.

        Parameters
        ----------
        t_eval : torch.Tensor of shape (batch_size, n_eval)
            Evaluation nodes.

        Returns
        -------
        interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Interpolated trajectories.
        deriv : torch.Tensor of shape (batch_size, n_eval, ndim)
            Piecewise-constant derivatives.
        """
        y_fit = self.y_fit_
        t_fit = self.t_fit_

        B, T1, D = y_fit.shape

        idx = torch.searchsorted(t_fit, t_eval, right=True) - 1
        idx = idx.clamp(0, T1 - 2)

        t0 = torch.gather(t_fit, 1, idx)
        t1 = torch.gather(t_fit, 1, idx + 1)

        y0 = torch.gather(y_fit, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        y1 = torch.gather(y_fit, 1, (idx + 1).unsqueeze(-1).expand(-1, -1, D))

        delta_t = (t1 - t0).unsqueeze(-1)
        alpha = ((t_eval - t0) / (t1 - t0)).unsqueeze(-1)
        alpha = torch.where(delta_t != 0, alpha, torch.zeros_like(alpha))

        interp = (1 - alpha) * y0 + alpha * y1
        deriv = torch.where(delta_t != 0, (y1 - y0) / delta_t, torch.zeros_like(y0))

        return interp, deriv
