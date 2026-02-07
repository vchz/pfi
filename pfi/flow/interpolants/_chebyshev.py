"""Chebyshev polynomial interpolants and regularization selection."""

import numpy as np
import torch

from ._base import BaseInterpolant


def chebyshev_basis_matrix(
    s,
    degree,
):
    """Compute first-kind Chebyshev basis values.

    Parameters
    ----------
    s : torch.Tensor of shape (batch_size, n_points)
        Scaled nodes in ``[-1, 1]``.
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    basis : torch.Tensor of shape (batch_size, n_points, degree + 1)
        Basis matrix ``[T_0(s), ..., T_degree(s)]``.
    """
    s = s.to(dtype=torch.float32)
    device = s.device

    v = [torch.ones_like(s, dtype=torch.float32, device=device), s]
    for n in range(2, degree + 1):
        tn = 2 * s * v[-1] - v[-2]
        v.append(tn)

    return torch.stack(v, dim=-1)


def chebyshev_U_basis_matrix(
    s,
    degree,
):
    """Compute second-kind Chebyshev basis values.

    Parameters
    ----------
    s : torch.Tensor of shape (batch_size, n_points)
        Scaled nodes in ``[-1, 1]``.
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    basis : torch.Tensor of shape (batch_size, n_points, degree)
        Basis matrix ``[U_0(s), ..., U_{degree-1}(s)]``.
    """
    s = s.to(dtype=torch.float32)
    device = s.device

    u = [torch.ones_like(s, dtype=torch.float32, device=device)]
    if degree >= 1:
        u.append(2 * s)

    for k in range(2, degree):
        u_next = 2 * s * u[-1] - u[-2]
        u.append(u_next)

    return torch.stack(u, dim=-1)


def batched_chebyshev_interpolate(
    t_points,
    x_points,
    degree=None,
    lambda_reg=0.0,
    penalty="none",
):
    """Fit batched Chebyshev polynomials and return evaluation callables.

    Parameters
    ----------
    t_points : torch.Tensor of shape (batch_size, n_points)
        Time nodes for fitting.
    x_points : torch.Tensor of shape (batch_size, n_points, ndim)
        Trajectory values at ``t_points``.
    degree : int or None, default=None
        Polynomial degree. If ``None``, uses ``n_points - 1``.
    lambda_reg : float, default=0.0
        Regularization weight.
    penalty : {'none', 'l2', 'velocity', 'curvature'}, default='none'
        Regularization profile on coefficients.

    Returns
    -------
    coeffs : torch.Tensor of shape (batch_size, degree + 1, ndim)
        Fitted Chebyshev coefficients.
    interpolant : callable
        Function mapping ``t_eval`` of shape ``(batch_size, n_eval)`` to
        interpolated values of shape ``(batch_size, n_eval, ndim)``.
    derivative : callable
        Function mapping ``t_eval`` of shape ``(batch_size, n_eval)`` to
        derivatives of shape ``(batch_size, n_eval, ndim)``.
    bounds : tuple of torch.Tensor
        Tuple ``(a, bmax)`` each of shape ``(batch_size, 1)`` used for scaling.
    """
    t_points = t_points.to(dtype=torch.float32)
    x_points = x_points.to(dtype=torch.float32)
    device = t_points.device
    lambda_reg = float(lambda_reg)

    bsz, n, _ = x_points.shape

    if degree is None:
        degree = n - 1

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
            uk = u_basis[:, :, k - 1:k]
            coeff_k = coeffs[:, k:k + 1, :]
            deriv += k * uk * coeff_k

        return dsdt * deriv

    return coeffs, interpolant, derivative, (a, bmax)


class ChebyshevInterpolant(BaseInterpolant):
    """Regularized Chebyshev interpolant for batched trajectories.

    Parameters
    ----------
    reg_ : float, default=0.01
        Curvature regularization weight.
    device : str or torch.device, default='cpu'
        Computation device.

    Attributes
    ----------
    reg_ : float
        Curvature regularization weight.
    y_fit_ : torch.Tensor of shape (batch_size, n_nodes, ndim)
        Training trajectories, set by `fit`.
    t_fit_ : torch.Tensor of shape (batch_size, n_nodes)
        Training time nodes, set by `fit`.
    p_ : callable
        Interpolant function set by `_fit`.
    p_prime_ : callable
        Derivative function set by `_fit`.
    """

    def __init__(
        self,
        reg_=0.01,
        device="cpu",
    ):
        super().__init__(
            device=device,
        )
        self.reg_ = reg_

    def _fit(
        self,
    ):
        """Fit Chebyshev coefficients from stored fit data.

        Returns
        -------
        self : ChebyshevInterpolant
            Fitted interpolant.
        """
        t_fit = self.t_fit_
        y_fit = self.y_fit_

        _, p, p_prime, _ = batched_chebyshev_interpolate(
            t_fit,
            y_fit,
            degree=t_fit.shape[1] - 1,
            lambda_reg=self.reg_,
            penalty="curvature",
        )

        self.p_ = p
        self.p_prime_ = p_prime
        return self

    def predict(
        self,
        t_eval,
    ):
        """Evaluate interpolated trajectories and derivatives.

        Parameters
        ----------
        t_eval : torch.Tensor of shape (batch_size, n_eval)
            Evaluation nodes.

        Returns
        -------
        x_interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Interpolated trajectories.
        dx_interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Time derivatives.
        """
        x_interp = self.p_(t_eval)
        dx_interp = self.p_prime_(t_eval)
        return x_interp, dx_interp


def select_best_lambda(
    batch_ot_samples,
    data_batch,
    eval_batch,
    device,
    lam_vals=None,
    rel_tol=0.8,
    verbose=True,
):
    """Select regularization strength from velocity-magnitude reduction.

    Parameters
    ----------
    batch_ot_samples : torch.Tensor of shape (batch_size, n_nodes, ndim)
        Training trajectories.
    data_batch : torch.Tensor of shape (batch_size, n_nodes)
        Fit nodes.
    eval_batch : torch.Tensor of shape (batch_size, n_eval)
        Evaluation nodes.
    device : str or torch.device
        Computation device.
    lam_vals : array-like of shape (n_lambdas,), default=None
        Candidate regularization values.
    rel_tol : float, default=0.8
        Minimum relative reduction threshold.
    verbose : bool, default=True
        If ``True``, print diagnostics.

    Returns
    -------
    best_lambda : float
        Selected regularization value.
    lam_arr : ndarray of shape (n_lambdas,)
        Candidate values evaluated.
    vel_mag : ndarray of shape (n_lambdas,)
        Mean squared derivative magnitudes for each candidate.
    """
    if lam_vals is None:
        lam_arr = np.array([0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0])
    else:
        lam_arr = np.array(lam_vals)

    vel_mag = np.zeros_like(lam_arr)

    for k in range(len(lam_arr)):
        interpolant = ChebyshevInterpolant(
            reg_=lam_arr[k],
            device=device,
        )
        interpolant.fit(data_batch, batch_ot_samples)
        _, dx_interp = interpolant.predict(eval_batch)
        vel_mag[k] = torch.mean(dx_interp ** 2).cpu().item()

    err0 = vel_mag[0]
    rel_err_drop = (err0 - vel_mag) / err0
    mask = rel_err_drop >= rel_tol

    if np.any(mask):
        idx = np.argmax(mask)
        best_lambda = lam_arr[idx]
    else:
        best_lambda = 0.01

    if verbose:
        print(f"[lambda-selection] Initial error: {err0:.4f}")
        print(f"[lambda-selection] Best lambda (>={rel_tol*100:.0f}% drop): {best_lambda:.4f}")
        print(f"[lambda-selection] Vel magnitudes: {vel_mag}")

    return best_lambda, lam_arr, vel_mag
