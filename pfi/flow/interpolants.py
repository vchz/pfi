"""Interpolation utilities for flow-matching solvers."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torchcubicspline


class Interpolant(ABC):
    """Abstract base class for trajectory interpolants."""

    @abstractmethod
    def fit(self, nodes_fit, dist):
        """Fit the interpolant on trajectory support nodes.

        Parameters
        ----------
        nodes_fit : torch.Tensor of shape (batch_size, n_nodes)
            Time nodes where trajectories are observed.
        dist : torch.Tensor of shape (batch_size, n_nodes, ndim)
            Trajectory values at ``nodes_fit``.

        Returns
        -------
        self : Interpolant
            Fitted interpolant instance.
        """

    @abstractmethod
    def predict(self, t_eval):
        """Evaluate interpolated values and time derivatives.

        Parameters
        ----------
        t_eval : torch.Tensor of shape (batch_size, n_eval)
            Time nodes where interpolation is evaluated.

        Returns
        -------
        interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Interpolated trajectory values.
        deriv : torch.Tensor of shape (batch_size, n_eval, ndim)
            Time derivatives at ``t_eval``.
        """


class LinearInterpolant(Interpolant):
    """Batched piecewise-linear interpolant."""

    def fit(self, nodes_fit, dist):
        """Store linear interpolation support points.

        Parameters
        ----------
        nodes_fit : torch.Tensor of shape (batch_size, n_nodes)
            Fit nodes.
        dist : torch.Tensor of shape (batch_size, n_nodes, ndim)
            Values at fit nodes.

        Returns
        -------
        self : LinearInterpolant
            Fitted interpolant.
        """
        self.t_fit_ = nodes_fit
        self.y_fit_ = dist
        return self

    def predict(self, t_eval):
        """Evaluate linear interpolation and piecewise-constant derivative.

        Parameters
        ----------
        t_eval : torch.Tensor of shape (batch_size, n_eval)
            Evaluation nodes.

        Returns
        -------
        interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Interpolated values.
        deriv : torch.Tensor of shape (batch_size, n_eval, ndim)
            Piecewise-constant time derivatives.
        """
        y_fit = self.y_fit_
        t_fit = self.t_fit_

        _, t1, ndim = y_fit.shape
        idx = torch.searchsorted(t_fit, t_eval, right=True) - 1
        idx = idx.clamp(0, t1 - 2)

        t0 = torch.gather(t_fit, 1, idx)
        t1v = torch.gather(t_fit, 1, idx + 1)
        y0 = torch.gather(y_fit, 1, idx.unsqueeze(-1).expand(-1, -1, ndim))
        y1 = torch.gather(y_fit, 1, (idx + 1).unsqueeze(-1).expand(-1, -1, ndim))

        delta_t = (t1v - t0).unsqueeze(-1)
        alpha = ((t_eval - t0) / (t1v - t0)).unsqueeze(-1)
        alpha = torch.where(delta_t != 0, alpha, torch.zeros_like(alpha))

        interp = (1 - alpha) * y0 + alpha * y1
        deriv = torch.where(delta_t != 0, (y1 - y0) / delta_t, torch.zeros_like(y0))
        return interp, deriv


def chebyshev_basis_matrix(s, degree):
    """Compute first-kind Chebyshev basis values.

    Parameters
    ----------
    s : torch.Tensor of shape (...,)
        Scaled coordinates in ``[-1, 1]``.
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    basis : torch.Tensor of shape (..., degree + 1)
        Values of ``T_k(s)`` for ``k = 0, ..., degree``.
    """
    s = s.to(dtype=torch.float32)
    device = s.device

    vals = [torch.ones_like(s, dtype=torch.float32, device=device), s]
    for n in range(2, degree + 1):
        vals.append(2 * s * vals[-1] - vals[-2])
    return torch.stack(vals, dim=-1)


def chebyshev_U_basis_matrix(s, degree):
    """Compute second-kind Chebyshev basis values.

    Parameters
    ----------
    s : torch.Tensor of shape (...,)
        Scaled coordinates in ``[-1, 1]``.
    degree : int
        Number of second-kind basis terms used for derivatives.

    Returns
    -------
    basis : torch.Tensor of shape (..., degree)
        Values of ``U_k(s)`` for ``k = 0, ..., degree - 1``.
    """
    s = s.to(dtype=torch.float32)
    device = s.device

    vals = [torch.ones_like(s, dtype=torch.float32, device=device)]
    if degree >= 1:
        vals.append(2 * s)
    for k in range(2, degree):
        vals.append(2 * s * vals[-1] - vals[-2])
    return torch.stack(vals, dim=-1)


def batched_chebyshev_interpolate(
    t_points,
    x_points,
    degree=None,
    lambda_reg=None,
    penalty="none",
):
    """Fit batched Chebyshev interpolants and derivative evaluators.

    Parameters
    ----------
    t_points : torch.Tensor of shape (batch_size, n_nodes)
        Fit nodes for each trajectory in the batch.
    x_points : torch.Tensor of shape (batch_size, n_nodes, ndim)
        Trajectory values at ``t_points``.
    degree : int or None, default=None
        Polynomial degree. If ``None``, use ``n_nodes - 1``.
    lambda_reg : float or None, default=None
        Regularization strength. ``None`` is treated as ``0.0``.
    penalty : {'none', 'l2', 'velocity', 'curvature'}, default='none'
        Regularization type for polynomial coefficients.

    Returns
    -------
    coeffs : torch.Tensor of shape (batch_size, degree + 1, ndim)
        Fitted Chebyshev coefficients.
    interpolant : callable
        Function mapping ``t_eval`` to interpolated values with shape
        ``(batch_size, n_eval, ndim)``.
    derivative : callable
        Function mapping ``t_eval`` to time derivatives with shape
        ``(batch_size, n_eval, ndim)``.
    bounds : tuple[torch.Tensor, torch.Tensor]
        ``(a, bmax)`` batch-wise min/max time bounds used for scaling.
    """
    if lambda_reg is None:
        lambda_reg = 0.0
    t_points = t_points.to(dtype=torch.float32)
    x_points = x_points.to(dtype=torch.float32)
    device = t_points.device
    lambda_reg = float(lambda_reg)

    bsz, n_nodes, _ = x_points.shape
    if degree is None:
        degree = n_nodes - 1

    a = t_points.min(dim=1, keepdim=True).values
    bmax = t_points.max(dim=1, keepdim=True).values
    s_points = (2 * t_points - (a + bmax)) / (bmax - a)

    v = chebyshev_basis_matrix(s_points, degree)
    vt = v.transpose(1, 2)

    powers = torch.arange(degree + 1, dtype=torch.float32, device=device)
    if penalty == "none" or lambda_reg == 0.0:
        r = torch.zeros_like(powers)
    elif penalty == "l2":
        r = torch.ones_like(powers)
        r[0] = 0
    elif penalty == "velocity":
        r = powers ** 2
        r[0] = 0
    elif penalty == "curvature":
        r = powers ** 4
        r[0] = 0
    else:
        raise ValueError("Invalid penalty type")

    r_mat = torch.diag(r).unsqueeze(0).expand(bsz, -1, -1).contiguous()
    lhs = vt @ v + lambda_reg * r_mat
    rhs = vt @ x_points
    coeffs = torch.linalg.solve(lhs, rhs).float()

    def interpolant(t_eval):
        t_eval = t_eval.to(dtype=torch.float32, device=device)
        s_eval = (2 * t_eval - (a + bmax)) / (bmax - a)
        basis = chebyshev_basis_matrix(s_eval, degree)
        return torch.einsum("bnd,bdc->bnc", basis, coeffs)

    def derivative(t_eval):
        t_eval = t_eval.to(dtype=torch.float32, device=device)
        s_eval = (2 * t_eval - (a + bmax)) / (bmax - a)
        dsdt = (2 / (bmax - a)).float().unsqueeze(-1)
        u_basis = chebyshev_U_basis_matrix(s_eval, degree)

        deriv = coeffs[:, 1:2, :].expand(-1, t_eval.shape[1], -1).clone()
        for k in range(2, degree + 1):
            deriv += k * u_basis[:, :, k - 1 : k] * coeffs[:, k : k + 1, :]
        return dsdt * deriv

    return coeffs, interpolant, derivative, (a, bmax)


class ChebyshevInterpolant(Interpolant):
    """Regularized Chebyshev interpolant for batched trajectories."""

    def __init__(self, reg=None):
        """Initialize a Chebyshev interpolant.

        Parameters
        ----------
        reg : float or None, default=None
            Curvature regularization strength used in ``fit``.
        """
        self.reg = reg

    def fit(self, nodes_fit, dist):
        """Fit Chebyshev coefficients from fit data.

        Parameters
        ----------
        nodes_fit : torch.Tensor of shape (batch_size, n_nodes)
            Fit nodes.
        dist : torch.Tensor of shape (batch_size, n_nodes, ndim)
            Trajectory values at fit nodes.

        Returns
        -------
        self : ChebyshevInterpolant
            Fitted interpolant.
        """
        self.t_fit_ = nodes_fit
        self.y_fit_ = dist
        _, p, p_prime, _ = batched_chebyshev_interpolate(
            nodes_fit,
            dist,
            degree=nodes_fit.shape[1] - 1,
            lambda_reg=self.reg,
            penalty="curvature",
        )
        self.p_ = p
        self.p_prime_ = p_prime
        return self

    def predict(self, t_eval):
        """Evaluate interpolated trajectories and derivatives.

        Parameters
        ----------
        t_eval : torch.Tensor of shape (batch_size, n_eval)
            Evaluation nodes.

        Returns
        -------
        interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Interpolated values.
        deriv : torch.Tensor of shape (batch_size, n_eval, ndim)
            Time derivatives.
        """
        return self.p_(t_eval), self.p_prime_(t_eval)


class SplineInterpolant(Interpolant):
    """Batched natural cubic spline interpolant."""

    def fit(self, nodes_fit, dist):
        """Fit natural cubic spline coefficients.

        Parameters
        ----------
        nodes_fit : torch.Tensor of shape (batch_size, n_nodes)
            Fit nodes.
        dist : torch.Tensor of shape (batch_size, n_nodes, ndim)
            Trajectory values at fit nodes.

        Returns
        -------
        self : SplineInterpolant
            Fitted interpolant.
        """
        self.t_fit_ = nodes_fit
        self.y_fit_ = dist
        t_fit_1d = nodes_fit[0] if nodes_fit.dim() == 2 else nodes_fit
        tempdist = dist[None, :, :, :]
        coeffs = torchcubicspline.interpolate.natural_cubic_spline_coeffs(t_fit_1d, tempdist)
        self.spline_ = torchcubicspline.interpolate.NaturalCubicSpline(coeffs)
        return self

    def predict(self, t_eval):
        """Evaluate spline interpolation and derivative.

        Parameters
        ----------
        t_eval : torch.Tensor of shape (batch_size, n_eval)
            Evaluation nodes.

        Returns
        -------
        interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Interpolated values.
        deriv : torch.Tensor of shape (batch_size, n_eval, ndim)
            Time derivatives.
        """
        t_eval_1d = t_eval[0] if t_eval.dim() == 2 else t_eval
        eval_ = self.spline_.evaluate(t_eval_1d)
        derv_ = self.spline_.derivative(t_eval_1d)

        eval_ = torch.permute(eval_, [2, 1, 0, 3]).squeeze()
        derv_ = torch.permute(derv_, [2, 1, 0, 3]).squeeze()
        eval_ = torch.permute(eval_, (1, 0, 2))
        derv_ = torch.permute(derv_, (1, 0, 2))
        return eval_, derv_


def select_best_lambda(
    batch_ot_samples,
    data_batch,
    eval_batch,
    lam_vals=None,
    rel_tol=0.8,
    verbose=True,
):
    """Select a Chebyshev regularization level from derivative smoothing.

    Parameters
    ----------
    batch_ot_samples : torch.Tensor of shape (batch_size, n_nodes, ndim)
        Trajectories used for interpolation fitting.
    data_batch : torch.Tensor of shape (batch_size, n_nodes)
        Fit nodes corresponding to ``batch_ot_samples``.
    eval_batch : torch.Tensor of shape (batch_size, n_eval)
        Evaluation nodes used to estimate derivative magnitude.
    lam_vals : array-like or None, default=None
        Candidate regularization strengths. If ``None``, a default grid is used.
    rel_tol : float, default=0.8
        Relative reduction threshold on mean squared derivative magnitude.
    verbose : bool, default=True
        If ``True``, print selection diagnostics.

    Returns
    -------
    best_lambda : float
        Selected regularization strength.
    lam_arr : ndarray of shape (n_lambdas,)
        Candidate lambda grid.
    vel_mag : ndarray of shape (n_lambdas,)
        Mean squared derivative magnitude for each lambda.
    """
    if lam_vals is None:
        lam_arr = np.array(
            [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]
        )
    else:
        lam_arr = np.array(lam_vals)

    vel_mag = np.zeros_like(lam_arr)
    for k, lam in enumerate(lam_arr):
        interpolant = ChebyshevInterpolant(reg=lam)
        interpolant.fit(data_batch, batch_ot_samples)
        _, dx_interp = interpolant.predict(eval_batch)
        vel_mag[k] = torch.mean(dx_interp ** 2).cpu().item()

    err0 = vel_mag[0]
    rel_err_drop = (err0 - vel_mag) / err0
    mask = rel_err_drop >= rel_tol
    best_lambda = lam_arr[np.argmax(mask)] if np.any(mask) else 0.01

    if verbose:
        print(f"[lambda-selection] Initial error: {err0:.4f}")
        print(f"[lambda-selection] Best lambda (>={rel_tol*100:.0f}% drop): {best_lambda:.4f}")
        print(f"[lambda-selection] Vel magnitudes: {vel_mag}")

    return best_lambda, lam_arr, vel_mag


__all__ = [
    "Interpolant",
    "LinearInterpolant",
    "ChebyshevInterpolant",
    "SplineInterpolant",
    "select_best_lambda",
]
