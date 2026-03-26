"""Simulation utilities for benchmark dynamical systems."""

import numpy as np


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
