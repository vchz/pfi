"""Closed-form PFM solver for models linear in their parameters (``OUFlow``).

Same steps as ``PFM_``: build the paths, interpolate them, and regress the model velocity on
the path velocities. For ``OUFlow`` the probability-flow velocity
``v(x, t) = -B x - lx x - D s(x, t)`` is linear in the drift matrix ``B``, so the regression
is a least-squares problem, solved exactly (with an optional ridge) instead of by stochastic
gradient descent.
"""

import numpy as np
import torch

from ..interpolants import ChebyshevInterpolant, select_best_lambda
from ..models import OUFlow
from ..paths import ExactOT
from ._pfm import _model_push


def _fit_drift(x, xdot, score, diff, lx=0.0, ridge=0.0):
    """Exact minimizer over ``B`` of the PFM loss for ``v(x) = -B x - lx x - D s(x)``.

    With ``z = -(xdot + lx x + D s)`` the loss is ``mean |z - B x|^2 + ridge |B|_F^2``, whose
    minimizer is ``B = mean(z x^T) (mean(x x^T) + ridge I)^{-1}``.

    Parameters
    ----------
    x, xdot, score : torch.Tensor of shape (m, ndim)
        Interpolated positions, their time derivatives, and the score at the positions,
        for all paths and evaluation times.
    diff : torch.Tensor of shape (ndim, ndim)
        Diffusion matrix ``D``.
    lx : float, default=0.0
        Linear degradation coefficient of the model.
    ridge : float, default=0.0
        Ridge penalty on ``B`` (time-averaged loss).

    Returns
    -------
    ndarray of shape (ndim, ndim)
        Fitted drift matrix.
    """
    x = x.to(dtype=torch.float64)
    target = -(xdot.to(torch.float64) + lx * x + score.to(torch.float64) @ diff.T)
    ndim = x.shape[1]

    zx = target.T @ x / x.shape[0]
    xx = x.T @ x / x.shape[0] + ridge * torch.eye(ndim, dtype=torch.float64, device=x.device)
    return torch.linalg.solve(xx, zx.T).T.cpu().numpy()


def _fixed_point(
    fit,
    n_rounds=1,
    anderson=0,
    safeguard=True,
    tol=None,
    history=None,
    verbose=False,
    name="pfm-linear",
):
    """Iterate ``B -> fit(B)`` from ``fit(None)``, optionally with safeguarded Anderson.

    Round 1 fits without a reference model; every further round fits with the current one.
    Anderson acceleration (memory ``anderson``) extrapolates ``B`` from the last rounds; with
    ``safeguard``, an extrapolated point is kept only if its residual ``|fit(B) - B|`` does not
    grow (and its evaluation succeeds), otherwise the iteration returns to the last accepted
    point, takes a plain step and clears the memory.

    Returns
    -------
    B : ndarray of shape (ndim, ndim)
        Last fitted matrix.
    changes : list of float
        Relative change ``|fit(B) - B| / |fit(B)|`` per round (NaN for round 1 and failed
        rounds).
    """
    B = fit(None)
    last = B
    changes = [float("nan")]
    if history is not None:
        history.append({"round": 1, "change": float("nan"), "B": B.copy()})

    X, G = [], []                 # Anderson memory: fitted matrices and residuals
    accepted = None               # last accepted point: (fit(B), residual norm)
    extrapolated = False          # whether B is an Anderson extrapolation
    for rnd in range(1, n_rounds):
        can_reject = safeguard and anderson and extrapolated and accepted is not None

        # evaluate the round map; a failed extrapolated point is a rejected step
        try:
            FB = fit(B)
            failed = not np.all(np.isfinite(FB))
        except RuntimeError:
            if not can_reject:
                raise
            failed = True
        if failed:
            if not can_reject:
                raise ValueError(f"{name}: non-finite fit in round {rnd + 1}.")
            changes.append(float("nan"))
            if verbose:
                print(f"{name}: round {rnd + 1}: Anderson step rejected (failed)", flush=True)
            B, X, G, extrapolated = accepted[0], [], [], False
            continue

        last = FB
        resid = (FB - B).ravel()
        change = float(np.linalg.norm(resid) / np.linalg.norm(FB))
        rejected = can_reject and np.linalg.norm(resid) > accepted[1]
        changes.append(change)
        if history is not None:
            history.append({"round": rnd + 1, "change": change, "B": FB.copy()})
        if verbose:
            note = " (Anderson step rejected)" if rejected else ""
            print(f"{name}: round {rnd + 1}: change {change:.3e}{note}", flush=True)

        # safeguard: back to the last accepted point, plain step, memory cleared
        if rejected:
            B, X, G, extrapolated = accepted[0], [], [], False
            continue
        if tol is not None and change < tol:
            break

        # next point: plain step F(B), or Anderson extrapolation from the memory
        accepted = (FB, np.linalg.norm(resid))
        X.append(FB.ravel())
        G.append(resid)
        X, G = X[-anderson - 1:], G[-anderson - 1:]
        extrapolated = bool(anderson and len(G) > 1)
        if extrapolated:
            dG = np.diff(np.array(G), axis=0).T
            dX = np.diff(np.array(X), axis=0).T
            gamma = np.linalg.lstsq(dG, resid, rcond=None)[0]
            B = (FB.ravel() - dX @ gamma).reshape(FB.shape)
        else:
            B = FB

    return last, changes


def _set_B(net, B):
    with torch.no_grad():
        net.net_.B.copy_(torch.as_tensor(B, dtype=net.net_.B.dtype, device=net.net_.B.device))


def PFM_LINEAR_(
    dist,
    times,
    net,
    interp=None,
    paths=None,
    growth_model=None,
    fac=4,
    t_eval=None,
    ridge=0.0,
    n_rounds=1,
    anderson=0,
    safeguard=True,
    tol=None,
    push_steps=8,
    history=None,
    verbose=True,
    generator=None,
):
    """Fit ``OUFlow`` in closed form by flow matching on path ensembles.

    Every round: (1) the model's flow is the reference dynamics (none in round 1),
    (2) the paths are built and sampled, (3) they are interpolated at the evaluation
    times, (4) the score is evaluated there, and (5) ``B`` is fitted by exact least squares.

    Parameters
    ----------
    dist : list of torch.Tensor
        Snapshot list where ``dist[k]`` has shape ``(n_k, ndim)``.
    times : torch.Tensor of shape (n_snaps,)
        Snapshot times.
    net : pfi.flow.models.OUFlow
        Model, without feature or batch normalization. Its drift matrix ``net.net_.B`` is
        set to the solution.
    interp : object or None, default=None
        Interpolant implementing ``fit`` and ``predict``. If ``None``,
        ``ChebyshevInterpolant()``.
    paths : pfi.flow.paths.PathEnsemble or None, default=None
        Path ensemble. If ``None``, ``ExactOT()``.
    growth_model : None
        Not supported (must be ``None``).
    fac : int, default=4
        Evaluation times per snapshot (uniform grid over the window, as in ``PFM_``).
    t_eval : array-like or None, default=None
        Explicit evaluation times (e.g. interval midpoints); overrides ``fac``.
    ridge : float, default=0.0
        Ridge penalty ``ridge * ||B||_F^2`` on the time-averaged loss.
    n_rounds : int, default=1
        Rounds of the iterated coupling (``1``: no iteration).
    anderson : int, default=0
        Anderson memory on ``B`` (``0``: plain iteration).
    safeguard : bool, default=True
        Keep an Anderson step only if the residual does not grow.
    tol : float or None, default=None
        Stop when the relative change of ``B`` is below ``tol``.
    push_steps : int, default=8
        Runge-Kutta 4 steps per snapshot interval when pushing with the model.
    history : list or None, default=None
        If a list, receives one dict per round: ``round``, ``change`` (relative change of
        ``B``, NaN for round 1) and ``B``.
    verbose : bool, default=True
        If ``True``, print one line per round.
    generator : torch.Generator or None, default=None
        RNG of the path sampling, reset at every round so that all rounds use the same
        draws (the round map is then deterministic). If ``None``, a generator seeded with 0.

    Returns
    -------
    net : pfi.flow.models.OUFlow
        Fitted model.
    growth_model : None
        Always ``None``.
    loss_hist : list of float
        Relative change of ``B`` per round (NaN for round 1).
    """
    if growth_model is not None:
        raise ValueError("pfm-linear does not support a growth model.")
    if not isinstance(net, OUFlow):
        raise ValueError("pfm-linear only supports OUFlow, whose velocity is linear in B.")
    if net.feature_norm or net.batch_norm:
        raise ValueError("pfm-linear needs an OUFlow without feature or batch normalization.")
    if interp is None:
        interp = ChebyshevInterpolant()
    if paths is None:
        paths = ExactOT()

    device = dist[0].device
    dtype = net.net_.B.dtype
    model_device = net.net_.B.device
    ndim = dist[0].shape[1]
    times = torch.as_tensor(times, device=device)
    if t_eval is None:
        t_eval = torch.linspace(times[0], times[-1], fac * times.shape[0], device=device)
    else:
        t_eval = torch.as_tensor(t_eval, device=device)
    diff = torch.as_tensor(net.D, dtype=torch.float64, device=device)
    if diff.dim() == 0:
        diff = diff * torch.eye(ndim, dtype=torch.float64, device=device)

    # the same random draws at every round, so that the round map B -> fit(B) is deterministic
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)
    rng_state = generator.get_state()

    def fit(B_model):
        # 1. reference dynamics: the model's flow with the current B (none in round 1)
        push = None
        if B_model is not None:
            _set_B(net, B_model)
            model_push = _model_push(net, times.to(device=model_device, dtype=dtype), push_steps)

            def push(x, k):
                y = model_push(x.to(device=model_device, dtype=dtype), k)
                return y.to(device=x.device, dtype=x.dtype)

        # 2. paths through the snapshots
        generator.set_state(rng_state)
        paths.fit(dist, push=push, generator=generator)
        trajs, _ = paths.sample(generator=generator)
        ntraj = trajs.shape[0]

        # 3. interpolate the paths: positions and velocities at the evaluation times
        nodes = times[None, :].repeat(ntraj, 1)
        grid = t_eval[None, :].repeat(ntraj, 1)
        if isinstance(interp, ChebyshevInterpolant) and interp.reg is None:
            interp.reg, _, _ = select_best_lambda(trajs, nodes, grid, verbose=verbose)
        interp.fit(nodes, trajs)
        x, xdot = interp.predict(grid)
        x = x.reshape(-1, ndim)
        xdot = xdot.reshape(-1, ndim)

        # 4. score of the model at the interpolated points
        t_col = grid.reshape(-1, 1).to(dtype=x.dtype)
        x_in = torch.cat([x, t_col], dim=1).to(device=model_device, dtype=dtype)
        with torch.no_grad():
            score = net.score(x_in).to(device=x.device)

        # 5. exact least-squares fit of B
        return _fit_drift(x, xdot, score, diff, lx=float(net.lx), ridge=ridge)

    B, loss_hist = _fixed_point(
        fit,
        n_rounds=n_rounds,
        anderson=anderson,
        safeguard=safeguard,
        tol=tol,
        history=history,
        verbose=verbose,
    )
    _set_B(net, B)
    return net, None, loss_hist


__all__ = ["PFM_LINEAR_"]
