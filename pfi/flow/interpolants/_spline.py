"""Natural cubic spline interpolant for trajectory data."""

import torch
import torchcubicspline

from ._base import BaseInterpolant


class SplineInterpolant(BaseInterpolant):
    """Batched natural cubic spline interpolant.

    Parameters
    ----------
    device : str or torch.device, default='cpu'
        Computation device inherited from `pfi.flow.interpolants.BaseInterpolant`.

    """

    def _fit(
        self,
    ):
        """Fit spline coefficients from stored training trajectories.

        Returns
        -------
        self : SplineInterpolant
            Fitted interpolant.
        """
        t_fit = self.t_fit_
        y_fit = self.y_fit_

        if t_fit.dim() == 2:
            t_fit_1d = t_fit[0]
        else:
            t_fit_1d = t_fit

        tempdist = y_fit[None, :, :, :]
        coeffs = torchcubicspline.interpolate.natural_cubic_spline_coeffs(t_fit_1d, tempdist)
        spline = torchcubicspline.interpolate.NaturalCubicSpline(coeffs)

        self.spline_ = spline
        return self

    def predict(self, t_eval):
        """Evaluate spline interpolation and derivative.

        Parameters
        ----------
        t_eval : torch.Tensor of shape (batch_size, n_eval) or (n_eval,)
            Evaluation nodes.

        Returns
        -------
        eval_ : torch.Tensor of shape (batch_size, n_eval, ndim)
            Interpolated trajectories.
        derv_ : torch.Tensor of shape (batch_size, n_eval, ndim)
            Derivative trajectories.
        """
        if t_eval.dim() == 2:
            t_eval_1d = t_eval[0]
        else:
            t_eval_1d = t_eval

        eval_ = self.spline_.evaluate(t_eval_1d)
        derv_ = self.spline_.derivative(t_eval_1d)

        eval_ = torch.permute(eval_, [2, 1, 0, 3]).squeeze()
        derv_ = torch.permute(derv_, [2, 1, 0, 3]).squeeze()

        eval_ = torch.permute(eval_, (1, 0, 2))
        derv_ = torch.permute(derv_, (1, 0, 2))

        return eval_, derv_
