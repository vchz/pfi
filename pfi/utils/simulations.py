"""Simulation utilities for benchmark dynamical systems."""

import numpy as np
import scipy.linalg as sla

def toggle_switch(
    x,
    model_params,
):
    """Evaluate the deterministic toggle-switch production term.

    Parameters
    ----------
    x : ndarray of shape (n_samples, 2)
        State values, where each row is ``[x1, x2]``.
    model_params : array-like of shape (7,)
        Model parameters ``[a1, a2, b1, b2, k1, k2, n]``.

    Returns
    -------
    f : ndarray of shape (n_samples, 2)
        Deterministic production term for each sample.
    """
    a1, a2, b1, b2, k1, k2, n = model_params
    xn = x ** n + 0.0
    f = np.zeros_like(x)
    f[:, 0] = (a1 * (xn[:, 0]) / (k1 ** n + xn[:, 0])) + b1 * (k1 ** n) / (k1 ** n + xn[:, 1])
    f[:, 1] = (a2 * (xn[:, 1]) / (k2 ** n + xn[:, 1])) + b2 * (k2 ** n) / (k2 ** n + xn[:, 0])
    return f

def g_rate(
    x1,
    x2,
    gr,
):
    """Compute the cell growth rate used in the toggle-switch simulator.

    Parameters
    ----------
    x1 : ndarray of shape (n_samples,)
        First coordinate of the state.
    x2 : ndarray of shape (n_samples,)
        Second coordinate of the state.
    gr : float
        Growth-rate scale.

    Returns
    -------
    rate : ndarray of shape (n_samples,)
        Growth rate for each sample.
    """
    return gr * (1.0 * (x2 ** 2) / (1 + x2 ** 2) + 0.0 * (x1 ** 2) / (1 + x1 ** 2))

def simulate_ornstein_uhlenbeck(
    Om,
    D,
    m0,
    S0,
    nsamples,
    ndim,
    Dt,
    K,
    dt=0.006,
):
    """Simulate snapshots from a linear Ornstein-Uhlenbeck process.

    Legacy: kept for backward compatibility with existing tests and examples. New code
    should use :func:`sample_ornstein_uhlenbeck`, which samples the exact Gaussian laws
    (no time-discretization error) for any initial covariance.

    Parameters
    ----------
    Om : ndarray of shape (ndim, ndim)
        Drift matrix.
    D : ndarray of shape (ndim, ndim)
        Diffusion matrix.
    m0 : ndarray of shape (ndim,)
        Mean of the initial Gaussian distribution.
    S0 : float
        Isotropic variance factor for the initial covariance ``S0 * I``.
    nsamples : int
        Number of particles sampled at each snapshot time.
    ndim : int
        State dimension.
    Dt : float
        Snapshot interval in simulation time.
    K : int
        Number of snapshots.
    dt : float, default=0.006
        Euler-Maruyama integration step.

    Returns
    -------
    samples : list of length K
        ``samples[k]`` is an ndarray of shape ``(nsamples, ndim)``.
    tt : ndarray of shape (K,)
        Snapshot times.
    """
    samples = []
    tt = np.zeros((K,))
    record = int(Dt / dt)

    for j in range(1, K + 1):
        traj_init = np.random.multivariate_normal(m0, S0 * np.eye(ndim), size=nsamples).T
        traj = traj_init.copy()

        for i in range(0, j * record):
            xi = np.random.normal(0, 1, (ndim, nsamples))
            traj = traj - (Om @ traj) * dt + (np.sqrt(2 * D) @ xi) * np.sqrt(dt)

        tt[j - 1] = dt * i
        samples.append(traj.T.copy())

    return samples, tt

def random_ou(ndim, rng, skew_scale=0.5):
    """Random stable Ornstein-Uhlenbeck process with symmetric and skew-symmetric parts.

    The drift matrix is ``B = S + A``, with ``S = Q diag(lambda) Q^T`` (``Q`` Haar-random
    orthogonal, ``lambda_i ~ U[0.1, 2]``, so ``B`` is stable) and
    ``A = skew_scale (G - G^T) / 2`` (``G_ij ~ N(0, 1)``). The diffusion is ``D = 0.5 I``.
    The initial law is far from stationarity: ``m0 ~ N(0, 9 I)`` and
    ``S0 = R diag(0.1 logspace(-1, 0, ndim)) R^T`` (``R`` Haar-random).

    Parameters
    ----------
    ndim : int
        State dimension.
    rng : numpy.random.Generator
        Random generator.
    skew_scale : float, default=0.5
        Scale of the skew-symmetric part; ``0`` gives a gradient (symmetric) drift.

    Returns
    -------
    B : ndarray of shape (ndim, ndim)
        Drift matrix, ``dx = -B x dt + sqrt(2 D) dW``.
    D : ndarray of shape (ndim, ndim)
        Diffusion matrix.
    m0 : ndarray of shape (ndim,)
        Initial mean.
    S0 : ndarray of shape (ndim, ndim)
        Initial covariance.
    """
    Q, _ = np.linalg.qr(rng.normal(size=(ndim, ndim)))
    S = Q @ np.diag(rng.uniform(0.1, 2.0, ndim)) @ Q.T
    G = rng.normal(size=(ndim, ndim))
    A = skew_scale * (G - G.T) / 2
    m0 = rng.normal(scale=3.0, size=ndim)
    R, _ = np.linalg.qr(rng.normal(size=(ndim, ndim)))
    S0 = R @ np.diag(0.1 * np.logspace(-1, 0, ndim)) @ R.T
    return S + A, 0.5 * np.eye(ndim), m0, S0


def ou_moments(B, D, m0, S0, t):
    """Exact mean and covariance of an Ornstein-Uhlenbeck process at time ``t``.

    For ``dx = -B x dt + sqrt(2 D) dW`` and ``x_0 ~ N(m0, S0)``:
    ``m_t = exp(-B t) m0`` and
    ``Sigma_t = exp(-B t) (S0 - S_inf) exp(-B t)^T + S_inf``, with the stationary
    covariance solving ``B S_inf + S_inf B^T = 2 D``.

    Parameters
    ----------
    B, D : ndarray of shape (ndim, ndim)
        Drift and diffusion matrices.
    m0 : ndarray of shape (ndim,)
        Initial mean.
    S0 : ndarray of shape (ndim, ndim)
        Initial covariance.
    t : float
        Time.

    Returns
    -------
    mean : ndarray of shape (ndim,)
        Mean at time ``t``.
    cov : ndarray of shape (ndim, ndim)
        Covariance at time ``t``.
    """
    B = np.asarray(B, dtype=float)
    S0 = np.asarray(S0, dtype=float)
    S_inf = sla.solve_continuous_lyapunov(B, 2 * np.asarray(D, dtype=float))
    E = sla.expm(-B * t)
    mean = E @ np.asarray(m0, dtype=float)
    cov = E @ (S0 - S_inf) @ E.T + S_inf
    return mean, cov


def sample_ornstein_uhlenbeck(B, D, m0, S0, times, nsamples, rng):
    """Sample independent snapshots of an Ornstein-Uhlenbeck process, exactly.

    Each snapshot is drawn from the exact Gaussian law at its time
    (:func:`ou_moments`), independently of the others, in the order of ``times``.

    Parameters
    ----------
    B, D : ndarray of shape (ndim, ndim)
        Drift and diffusion matrices.
    m0 : ndarray of shape (ndim,)
        Initial mean.
    S0 : ndarray of shape (ndim, ndim)
        Initial covariance.
    times : array-like of shape (K,)
        Snapshot times.
    nsamples : int
        Number of cells per snapshot.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    samples : list of length K
        ``samples[k]`` is an ndarray of shape ``(nsamples, ndim)``.
    times : ndarray of shape (K,)
        Snapshot times.
    """
    times = np.asarray(times, dtype=float)
    samples = []
    for t in times:
        mean, cov = ou_moments(B, D, m0, S0, t)
        samples.append(rng.multivariate_normal(mean, cov, size=nsamples))
    return samples, times


def ou_pfm_limit(
    B,
    D,
    m0,
    S0,
    times,
    paths,
    interp,
    lx=0.0,
    fac=4,
    t_eval=None,
    ridge=0.0,
    n_rounds=1,
    anderson=0,
    safeguard=True,
    tol=None,
    push_steps=8,
    history=None,
    verbose=False,
):
    """Large-n limit of the closed-form PFM fit (``pfm-linear``) for an Ornstein-Uhlenbeck process.

    The exact answer of ``PFM_LINEAR_`` when every snapshot is its exact Gaussian law: the
    sample averages of the least-squares fit become exact expectations. Nothing is fitted on
    data; the inputs are the true process and the method (path ensemble, interpolant).

    Every round:

    1. reference dynamics: the flow of the current model, ``v(x, t) = -B x - lx x - D s_t(x)``
       with the exact score ``s_t(x) = -Sigma_t^{-1} (x - m_t)`` (none in round 1);
    2. paths: the path ensemble fitted on the exact laws, which returns the exact means and
       covariances of its nodes (Gaussian ensembles and ``Independent``);
    3. interpolation: the interpolant is linear in the nodes, ``I(t) = sum_j w_j(t) x_j``, so
       ``E[I I^T]`` and ``E[dI I^T]`` follow from the node moments;
    4. score: ``E[s(I) I^T] = -Sigma_t^{-1} (E[I I^T] - m_t E[I]^T)``;
    5. exact least-squares fit of the drift matrix.

    Parameters
    ----------
    B, D : ndarray of shape (ndim, ndim)
        Drift and diffusion matrices of the process, ``dx = -B x dt + sqrt(2 D) dW``.
    m0 : ndarray of shape (ndim,)
        Initial mean.
    S0 : ndarray of shape (ndim, ndim)
        Initial covariance.
    times : array-like of shape (K,)
        Snapshot times.
    paths : pfi.flow.paths.PathEnsemble
        Path ensemble accepting exact laws (``GaussianOT``, ``GaussianDebiasedSinkhorn``,
        ``Independent``).
    interp : object
        Interpolant implementing ``fit`` and ``predict`` (linear in the node values).
    lx : float, default=0.0
        Linear degradation coefficient of the model.
    fac : int, default=4
        Evaluation times per snapshot (uniform grid over the window, as in ``PFM_``).
    t_eval : array-like or None, default=None
        Explicit evaluation times (e.g. interval midpoints); overrides ``fac``.
    ridge : float, default=0.0
        Ridge penalty on the drift matrix (time-averaged loss).
    n_rounds, anderson, safeguard, tol, push_steps, history, verbose
        Iteration settings, as in ``PFM_LINEAR_``.

    Returns
    -------
    ndarray of shape (ndim, ndim)
        Limit of the fitted drift matrix.
    """
    import torch

    from ..flow.solvers._pfm_linear import _fixed_point

    times = np.asarray(times, dtype=float)
    ndim = np.asarray(B).shape[0]
    D = np.asarray(D, dtype=float)
    if t_eval is None:
        t_eval = np.linspace(times[0], times[-1], fac * len(times))
    t_eval = np.asarray(t_eval, dtype=float)

    # exact snapshot laws, and the exact score s_t(x) = c_t - P_t x at any time
    laws = [tuple(torch.tensor(a) for a in ou_moments(B, D, m0, S0, t)) for t in times]

    def score_coeffs(t):
        mean, cov = ou_moments(B, D, m0, S0, t)
        prec = np.linalg.inv(cov)
        return prec @ mean, prec

    # interpolant weights on the nodes: x(t_e) = sum_j w[e, j] x_j (interpolate unit vectors;
    # two identical channels, since SplineInterpolant.predict squeezes a dimension of size 1)
    n_nodes = len(times)
    nodes = torch.tensor(times, dtype=torch.float32)[None, :].repeat(n_nodes, 1)
    unit = torch.eye(n_nodes)[:, :, None].repeat(1, 1, 2)
    interp.fit(nodes, unit)
    grid = torch.tensor(t_eval, dtype=torch.float32)[None, :].repeat(n_nodes, 1)
    values, derivs = interp.predict(grid)
    w = values[:, :, 0].T.double().numpy()
    w_dot = derivs[:, :, 0].T.double().numpy()

    eval_coeffs = [score_coeffs(t) for t in t_eval]
    stage_coeffs = []
    for k in range(n_nodes - 1):
        h = (times[k + 1] - times[k]) / push_steps
        stage_coeffs.append(
            [score_coeffs(times[k] + 0.5 * i * h) for i in range(2 * push_steps + 1)]
        )

    def model_push(B_model):
        """Flow of the model velocity over one interval (Runge-Kutta 4, as ``PFM_``)."""
        drift = -np.asarray(B_model) - lx * np.eye(ndim)

        def velocity(x, coeffs):
            offset, prec = coeffs
            return x @ (drift + D @ prec).T - offset @ D.T

        def push(x, k):
            y = x.detach().double().cpu().numpy()
            h = (times[k + 1] - times[k]) / push_steps
            stages = stage_coeffs[k]
            for i in range(push_steps):
                k1 = velocity(y, stages[2 * i])
                k2 = velocity(y + 0.5 * h * k1, stages[2 * i + 1])
                k3 = velocity(y + 0.5 * h * k2, stages[2 * i + 1])
                k4 = velocity(y + h * k3, stages[2 * i + 2])
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return torch.as_tensor(y, dtype=x.dtype)

        return push

    def fit(B_model):
        # 1. reference dynamics (none in round 1)
        push = None if B_model is None else model_push(B_model)

        # 2. exact moments of the path nodes
        paths.fit(laws, push=push)
        means, covs = paths.moments()
        means = means.double().numpy()
        second = covs.double().numpy() + means[:, None, :, None] * means[None, :, None, :]

        # 3. exact moments of the interpolated positions and velocities
        e_x = w @ means
        e_xx = np.einsum("ej,ek,jkab->eab", w, w, second)
        e_dxx = np.einsum("ej,ek,jkab->eab", w_dot, w, second)

        # 4. and 5. exact least squares: B = mean_e E[z x^T] (mean_e E[x x^T] + ridge I)^-1,
        # with z = -(dI + lx I + D s(I))
        zx = np.zeros((ndim, ndim))
        xx = ridge * np.eye(ndim)
        for e, (offset, prec) in enumerate(eval_coeffs):
            e_sx = np.outer(offset, e_x[e]) - prec @ e_xx[e]
            zx -= (e_dxx[e] + lx * e_xx[e] + D @ e_sx) / len(t_eval)
            xx += e_xx[e] / len(t_eval)
        return np.linalg.solve(xx, zx.T).T

    B_hat, _ = _fixed_point(
        fit,
        n_rounds=n_rounds,
        anderson=anderson,
        safeguard=safeguard,
        tol=tol,
        history=history,
        verbose=verbose,
        name="ou_pfm_limit",
    )
    return B_hat


def simulate_toggle_switch(
    model_params,
    vol,
    gr,
    nsamples,
    ndim,
    Dt,
    K,
    dt=0.01,
):
    """Simulate snapshots from stochastic toggle-switch dynamics.

    Parameters
    ----------
    model_params : array-like of shape (7,)
        Toggle-switch parameters ``[a1, a2, b1, b2, k1, k2, n]``.
    vol : float
        System volume scaling the stochastic term.
    gr : float
        Growth-rate scale. Growth is active only when ``gr > 0``.
    nsamples : int
        Number of particles sampled at the initial time.
    ndim : int
        State dimension.
    Dt : float
        Snapshot interval in simulation time.
    K : int
        Number of snapshots.
    dt : float, default=0.01
        Euler-Maruyama integration step.

    Returns
    -------
    samples_full : list of length K
        ``samples_full[k]`` is an ndarray of shape ``(n_k, ndim)``, where
        ``n_k`` can increase across snapshots when growth is active.
    tt : ndarray of shape (K,)
        Snapshot times.
    """
    lx = 1.0
    tt = np.zeros(K)
    samples_full = []
    record = int(Dt / dt)

    for j in range(1, K + 1):
        traj = np.random.uniform(0.25, 0.75, (ndim, nsamples))

        for _ in range(0, j * record):
            fval = toggle_switch(traj.T, model_params).T
            noise = np.sqrt(fval + lx * traj) * np.sqrt(dt) * np.random.normal(0, 1, traj.shape)
            xnew = traj + dt * (fval - lx * traj) + (1 / np.sqrt(vol)) * noise
            traj = np.where(xnew < 0, traj, xnew)

            if gr > 0:
                growth_probs = g_rate(traj[0], traj[1], gr) * dt
                divide_flags = np.random.rand(traj.shape[1]) < growth_probs
                new_cells = traj[:, divide_flags]
                if new_cells.shape[1] > 0:
                    traj = np.concatenate([traj, new_cells], axis=1)

        tt[j - 1] = j * Dt
        samples_full.append(traj.T.copy())

    return samples_full, tt
