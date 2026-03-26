"""Unbalanced flow-matching solvers used for the UFM paper setup."""

import numpy as np
import torch
from tqdm import tqdm

from ...couplings import pairwise_OT
from ...interpolants import ChebyshevInterpolant, LinearInterpolant, select_best_lambda
from .._pfm import compute_conditional_distributions


def compute_conditional_distributions_unbalanced(
    interp,
    growth_interp,
    psi_traj,
    dist,
    batch_size,
    nodes_fit,
    nodes_eval,
    sigma=0.001,
):
    """Build weighted training pairs for unbalanced flow matching.

    Parameters
    ----------
    interp : object
        Interpolant for state trajectories.
    growth_interp : object
        Interpolant for ``log(psi_t)`` trajectories.
    psi_traj : torch.Tensor of shape (n_snaps, n_samples)
        Per-trajectory endpoint weights.
    dist : torch.Tensor of shape (n_snaps, n_samples, ndim)
        Sampled trajectories at snapshot nodes.
    batch_size : int
        Number of sampled trajectories per optimization step.
    nodes_fit : torch.Tensor of shape (batch_size, n_snaps)
        Interpolant fit nodes.
    nodes_eval : torch.Tensor of shape (batch_size, n_eval)
        Interpolant evaluation nodes.
    sigma : float, default=0.001
        Standard deviation of additive Gaussian jitter on interpolated inputs.

    Returns
    -------
    xtrain : torch.Tensor of shape (batch_size * n_eval, ndim + 1)
        Interpolated states with time appended.
    ytrain : torch.Tensor of shape (batch_size * n_eval, ndim)
        Interpolated drift targets.
    wtrain : torch.Tensor of shape (batch_size * n_eval,)
        Interpolated mass weights ``psi_t``.
    gtrain : torch.Tensor of shape (batch_size * n_eval,)
        Interpolated growth targets ``d/dt log(psi_t)``.
    """
    nsamples = dist.shape[1]
    n_eval = nodes_eval.shape[1]
    ndim = dist.shape[2]
    device = dist.device
    dtype = dist.dtype

    xtrain = torch.zeros((batch_size * n_eval, ndim + 1), dtype=dtype, device=device)
    ytrain = torch.zeros((batch_size * n_eval, ndim), dtype=dtype, device=device)

    xind = torch.randint(0, nsamples, (batch_size,), device=device)
    dist_batch = torch.permute(dist[:, xind, :], (1, 0, 2))

    interp.fit(nodes_fit, dist_batch)
    x_interp, dx_interp = interp.predict(nodes_eval)

    psi_batch = psi_traj[:, xind].T[:, :, None]
    log_psi_batch = torch.log(psi_batch.clamp_min(torch.finfo(dtype).tiny))
    growth_interp.fit(nodes_fit, log_psi_batch)
    log_psi_t, dlog_psi_t = growth_interp.predict(nodes_eval)

    xtrain[:, ndim] = nodes_eval.reshape(batch_size * n_eval)
    mut = x_interp.reshape(batch_size * n_eval, ndim)
    xtrain[:, :ndim] = mut + sigma * torch.randn_like(mut)
    ytrain[:, :ndim] = dx_interp.reshape(batch_size * n_eval, ndim)

    log_psi = log_psi_t.reshape(batch_size * n_eval)
    wtrain = torch.exp(log_psi)
    gtrain = dlog_psi_t.reshape(batch_size * n_eval)
    return xtrain, ytrain, wtrain, gtrain


def UFM_OT_(
    dist,
    times,
    net,
    interp=None,
    growth_model=None,
    bs=None,
    fac=1,
    nb=1,
    n_epochs=2000,
    lr=1e-3,
    alpha_ann=0.5,
    scheduler_kwargs=None,
    ot="exact",
    reg=0.1,
    verbose=True,
):
    """Two-phase FM: optimize growth first, then drift with learned masses.

    Parameters
    ----------
    dist : list of torch.Tensor
        Snapshot list where ``dist[k]`` has shape ``(n_k, ndim)``.
    times : torch.Tensor of shape (n_snaps,)
        Snapshot times.
    net : torch.nn.Module
        Drift model receiving ``(x, t)`` as input.
    interp : object or None, default=None
        Interpolant implementing ``fit`` and ``predict``. If ``None``,
        ``LinearInterpolant()`` is used.
    growth_model : torch.nn.Module or None, default=None
        Optional growth model receiving spatial coordinates only.
    bs : int or None, default=None
        Number of sampled trajectories per optimization step. If ``None``,
        use all sampled trajectories.
    fac : int, default=1
        Number of interpolation nodes per snapshot interval multiplier.
    nb : int, default=1
        Number of OT minibatches.
    n_epochs : int, default=2000
        Number of optimizer iterations for each phase.
    lr : float, default=1e-3
        Learning rate for Adam optimizers.
    alpha_ann : float, default=0.5
        Unused placeholder kept for API stability.
    scheduler_kwargs : dict or None, default=None
        Optional kwargs passed to ``MultiStepLR``.
    ot : str, default="exact"
        Pairwise OT backend name.
    reg : float, default=0.1
        Entropic regularization for OT solver.
    verbose : bool, default=True
        If ``True``, display progress bars.

    Returns
    -------
    drift_net : torch.nn.Module
        Trained drift model.
    growth_model : torch.nn.Module or None
        Trained growth model, or ``None``.
    loss_hist : list
        Two-list history ``[drift_loss_hist, growth_loss_hist]``.
    """
    _ = alpha_ann
    if interp is None:
        interp = LinearInterpolant()

    nsamples = min(s.shape[0] for s in dist)
    dist_tensor = torch.stack(
        [s[torch.randperm(s.shape[0], device=s.device)[:nsamples]] for s in dist],
        dim=0,
    )
    device = dist_tensor.device

    x_mean = dist_tensor.mean(dim=(0, 1))
    x_std = dist_tensor.std(dim=(0, 1))
    t_mean = times.mean().to(device=device, dtype=x_mean.dtype)
    t_std = times.std().to(device=device, dtype=x_std.dtype)
    net.set_scales(torch.cat([x_mean, t_mean[None]]), torch.cat([x_std, t_std[None]]))

    batch_ot_samples, _ = pairwise_OT(dist, nb=nb, reg=reg, reg_m=None, method=ot)
    nsnaps, nsamples, ndim = batch_ot_samples.shape

    mass_vec = torch.ones((nsnaps,), dtype=dist_tensor.dtype, device=device)
    for k in range(1, nsnaps):
        mass_vec[k] = dist[k].shape[0] / dist[0].shape[0]

    uniform_kind = torch.linspace(
        times[0], times[-1], fac * times.shape[0], dtype=dist_tensor.dtype, device=device
    )
    data_nodes = times[None, :].repeat(nsamples, 1)
    uniform_nodes = uniform_kind.expand(nsamples, -1)

    batch_size = nsamples if bs is None else bs
    uniform_batch = uniform_kind.expand(batch_size, -1)
    data_batch = times[None, :].expand(batch_size, -1)

    batch_ot_samples_ = torch.permute(batch_ot_samples, (1, 0, 2))
    if isinstance(interp, ChebyshevInterpolant):
        best_lam, _, _ = select_best_lambda(
            batch_ot_samples_,
            data_nodes,
            uniform_nodes,
            verbose=verbose,
        )
        interp.reg = best_lam

    drift_net = net.to(device)
    zero_growth = growth_model is None
    if not zero_growth:
        growth_model = growth_model.to(device)

    scheduler_kwargs = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
    default_sched = {"milestones": [1000, 1500, 8000, 15000], "gamma": 0.1}
    default_sched.update(scheduler_kwargs)

    optimizer_growth = None
    scheduler_growth = None
    if not zero_growth:
        optimizer_growth = torch.optim.Adam(growth_model.parameters(), lr=lr)
        scheduler_growth = torch.optim.lr_scheduler.MultiStepLR(optimizer_growth, **default_sched)

    optimizer_drift = torch.optim.Adam(drift_net.parameters(), lr=lr)
    scheduler_drift = torch.optim.lr_scheduler.MultiStepLR(optimizer_drift, **default_sched)

    dt = uniform_kind[1] - uniform_kind[0]
    drift_loss_hist = []
    growth_loss_hist = []

    if not zero_growth:
        pbar_g = tqdm(range(n_epochs), desc="ufm_ot-growth", dynamic_ncols=True, disable=not verbose)
        for _ in pbar_g:
            xtrain, _ = compute_conditional_distributions(
                interp, batch_ot_samples, batch_size, data_batch, uniform_batch, device=device, sigma=0.001
            )
            optimizer_growth.zero_grad()
            growth_eval = growth_model(xtrain[:, :ndim]).reshape(batch_size, fac * nsnaps)
            log_mass = torch.zeros_like(growth_eval)
            log_mass[:, 0] = np.log(1.0 / batch_size)
            for k in range(1, fac * nsnaps):
                log_mass[:, k] = log_mass[:, k - 1] + 0.5 * dt * (growth_eval[:, k] + growth_eval[:, k - 1])
            mass_traj = torch.exp(log_mass)
            growth_loss = torch.sum((torch.sum(mass_traj, axis=0)[::fac] - mass_vec) ** 2)
            growth_loss.backward()
            optimizer_growth.step()
            scheduler_growth.step()
            growth_loss_hist.append(growth_loss.item())
            if verbose:
                pbar_g.set_postfix(growth=f"{growth_loss.item():.3e}", lr=f"{optimizer_growth.param_groups[0]['lr']:.2e}")
    else:
        growth_loss_hist = [0.0] * n_epochs

    pbar_d = tqdm(range(n_epochs), desc="ufm_ot-drift", dynamic_ncols=True, disable=not verbose)
    for _ in pbar_d:
        xtrain, ytrain = compute_conditional_distributions(
            interp, batch_ot_samples, batch_size, data_batch, uniform_batch, device=device, sigma=0.001
        )

        if zero_growth:
            mass_traj = torch.ones((batch_size, fac * nsnaps), dtype=xtrain.dtype, device=device)
            growth_value = 0.0
        else:
            with torch.no_grad():
                growth_eval = growth_model(xtrain[:, :ndim]).reshape(batch_size, fac * nsnaps)
                log_mass = torch.zeros_like(growth_eval)
                log_mass[:, 0] = np.log(1.0 / batch_size)
                for k in range(1, fac * nsnaps):
                    log_mass[:, k] = log_mass[:, k - 1] + 0.5 * dt * (growth_eval[:, k] + growth_eval[:, k - 1])
                mass_traj = torch.exp(log_mass)
            growth_value = growth_loss_hist[-1] if growth_loss_hist else 0.0

        optimizer_drift.zero_grad()
        drift_pred = drift_net(xtrain)
        res = (drift_pred - ytrain).pow(2).sum(dim=1).reshape(batch_size, fac * nsnaps)
        drift_loss = torch.sum(torch.mean(mass_traj * res, axis=0))
        drift_loss.backward()
        optimizer_drift.step()
        scheduler_drift.step()

        drift_loss_hist.append(drift_loss.item())
        if verbose:
            lr_drift = optimizer_drift.param_groups[0]["lr"]
            pbar_d.set_postfix(drift=f"{drift_loss.item():.3e}", growth=f"{growth_value:.3e}", lr=f"{lr_drift:.2e}")

    return drift_net, growth_model, [drift_loss_hist, growth_loss_hist]


def UFM_UOT_(
    dist,
    times,
    net,
    interp=None,
    growth_model=None,
    growth_interp=None,
    bs=None,
    fac=1,
    nb=1,
    n_epochs=2000,
    lr=1e-3,
    scheduler_kwargs=None,
    reg=0.1,
    reg_m=10.0,
    ot="sinkhorn",
    verbose=True,
):
    """Train split drift/growth objectives with unbalanced pairwise OT.

    Parameters
    ----------
    dist : list of torch.Tensor
        Snapshot list where ``dist[k]`` has shape ``(n_k, ndim)``.
    times : torch.Tensor of shape (n_snaps,)
        Snapshot times.
    net : torch.nn.Module
        Drift model receiving ``(x, t)`` as input.
    interp : object or None, default=None
        Interpolant for state trajectories. If ``None``,
        ``LinearInterpolant()`` is used.
    growth_model : torch.nn.Module or None, default=None
        Optional growth model receiving spatial coordinates only.
    growth_interp : object or None, default=None
        Interpolant used for ``log(psi_t)``.
    bs : int or None, default=None
        Number of sampled trajectories per optimization step. If ``None``,
        use all sampled trajectories.
    fac : int, default=1
        Number of interpolation nodes per snapshot interval multiplier.
    nb : int, default=1
        Number of OT minibatches.
    n_epochs : int, default=2000
        Number of optimizer iterations.
    lr : float, default=1e-3
        Learning rate for Adam optimizers.
    scheduler_kwargs : dict or None, default=None
        Optional kwargs passed to ``MultiStepLR``.
    reg : float, default=0.1
        Entropic regularization for OT solver.
    reg_m : float, default=10.0
        Mass regularization for unbalanced OT. Ignored if ``growth_model`` is ``None``.
    ot : str, default="sinkhorn"
        Pairwise OT backend name.
    verbose : bool, default=True
        If ``True``, display progress bars.

    Returns
    -------
    drift_net : torch.nn.Module
        Trained drift model.
    growth_net : torch.nn.Module or None
        Trained growth model, or ``None``.
    loss_hist : list
        Two-list history ``[flow_loss_hist, growth_loss_hist]``.
    """
    if interp is None:
        interp = LinearInterpolant()
    no_growth = growth_model is None
    if no_growth:
        reg_m = None

    nsamples = min(s.shape[0] for s in dist)
    dist_tensor = torch.stack(
        [s[torch.randperm(s.shape[0], device=s.device)[:nsamples]] for s in dist],
        dim=0,
    )
    device = dist_tensor.device

    x_mean = dist_tensor.mean(dim=(0, 1))
    x_std = dist_tensor.std(dim=(0, 1))
    t_mean = times.mean().to(device=device, dtype=x_mean.dtype)
    t_std = times.std().to(device=device, dtype=x_std.dtype)
    net.set_scales(torch.cat([x_mean, t_mean[None]]), torch.cat([x_std, t_std[None]]))

    batch_ot_samples, psi_traj = pairwise_OT(dist, nb=nb, reg=reg, reg_m=reg_m, method=ot)
    _, nsamples, ndim = batch_ot_samples.shape

    uniform_kind = torch.linspace(
        times[0],
        times[-1],
        fac * times.shape[0],
        dtype=dist_tensor.dtype,
        device=device,
    )
    data_nodes = times[None, :].repeat(nsamples, 1)
    uniform_nodes = uniform_kind.expand(nsamples, -1)

    batch_ot_samples_ = torch.permute(batch_ot_samples, (1, 0, 2))
    if isinstance(interp, ChebyshevInterpolant):
        best_lam, _, _ = select_best_lambda(
            batch_ot_samples_,
            data_nodes,
            uniform_nodes,
            device,
            verbose=verbose,
        )
        interp.reg = best_lam

    batch_size = nsamples if bs is None else bs
    data_batch_nodes = times[None, :].repeat(batch_size, 1)
    uniform_batch_nodes = uniform_kind.expand(batch_size, -1)

    drift_net = net.to(device)
    growth_net = None if no_growth else growth_model.to(device)
    growth_interp = LinearInterpolant() if growth_interp is None else growth_interp

    optimizer_drift = torch.optim.Adam(drift_net.parameters(), lr=lr)
    scheduler_kwargs = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
    default_sched = {"milestones": [1000, 1500, 8000, 15000], "gamma": 0.1}
    default_sched.update(scheduler_kwargs)
    scheduler_drift = torch.optim.lr_scheduler.MultiStepLR(optimizer_drift, **default_sched)

    optimizer_growth = None
    scheduler_growth = None
    if growth_net is not None:
        optimizer_growth = torch.optim.Adam(growth_net.parameters(), lr=lr)
        scheduler_growth = torch.optim.lr_scheduler.MultiStepLR(optimizer_growth, **default_sched)

    flow_loss_hist = []
    growth_loss_hist = []
    pbar = tqdm(range(n_epochs), desc="ufm_uot", dynamic_ncols=True, disable=not verbose)
    for _ in pbar:
        xtrain, ytrain, wtrain, gtrain = compute_conditional_distributions_unbalanced(
            interp,
            growth_interp,
            psi_traj,
            batch_ot_samples,
            batch_size,
            data_batch_nodes,
            uniform_batch_nodes,
            sigma=0.001,
        )

        optimizer_drift.zero_grad()
        drift_pred = drift_net(xtrain)
        loss_flow = torch.mean(wtrain * (drift_pred - ytrain).pow(2).sum(dim=1))
        loss_flow.backward()
        optimizer_drift.step()
        scheduler_drift.step()

        if growth_net is None:
            growth_value = 0.0
        else:
            optimizer_growth.zero_grad()
            growth_pred = growth_net(xtrain[:, :ndim]).squeeze(-1)
            loss_growth = torch.mean(wtrain * (growth_pred - gtrain).pow(2))
            loss_growth.backward()
            optimizer_growth.step()
            scheduler_growth.step()
            growth_value = loss_growth.item()

        flow_loss_hist.append(loss_flow.item())
        growth_loss_hist.append(growth_value)

        if verbose:
            lr_drift = optimizer_drift.param_groups[0]["lr"]
            if growth_net is None:
                pbar.set_postfix(flow=f"{loss_flow.item():.3e}", lr=f"{lr_drift:.2e}")
            else:
                lr_growth = optimizer_growth.param_groups[0]["lr"]
                pbar.set_postfix(
                    flow=f"{loss_flow.item():.3e}",
                    growth=f"{growth_value:.3e}",
                    lr_f=f"{lr_drift:.2e}",
                    lr_g=f"{lr_growth:.2e}",
                )

    return drift_net, growth_net, [flow_loss_hist, growth_loss_hist]


__all__ = [
    "UFM_UOT_",
    "UFM_OT_",
    "compute_conditional_distributions_unbalanced",
]
