"""Flow-matching PFM solver."""

import numpy as np
import torch
from tqdm import tqdm

from ...utils.nns import loss_grad_std
from ..couplings import pairwise_OT
from ..interpolants import ChebyshevInterpolant, LinearInterpolant, select_best_lambda


def compute_conditional_distributions(
    interp,
    dist,
    batch_size,
    nodes_fit,
    nodes_eval,
    device="cpu",
    sigma=0.001,
):
    """Build conditional training pairs for drift regression.

    Parameters
    ----------
    interp : object
        Interpolant implementing ``fit`` and ``predict``.
    dist : torch.Tensor of shape (n_snaps, n_samples, ndim)
        Trajectory support points at snapshot times.
    batch_size : int
        Number of sampled trajectories per optimization step.
    nodes_fit : torch.Tensor of shape (batch_size, n_snaps)
        Nodes used to fit interpolants.
    nodes_eval : torch.Tensor of shape (batch_size, n_eval)
        Nodes where interpolants are evaluated.
    device : str or torch.device, default="cpu"
        Device for generated training tensors.
    sigma : float, default=0.001
        Standard deviation of additive Gaussian jitter on interpolated inputs.

    Returns
    -------
    xtrain : torch.Tensor of shape (batch_size * n_eval, ndim + 1)
        Interpolated states with time appended.
    ytrain : torch.Tensor of shape (batch_size * n_eval, ndim)
        Time derivatives of interpolated states.
    """
    nsamples = dist.shape[1]
    n_eval = nodes_eval.shape[1]
    ndim = dist.shape[2]
    dtype = dist.dtype

    xtrain = torch.zeros((batch_size * n_eval, ndim + 1), dtype=dtype, device=device)
    ytrain = torch.zeros((batch_size * n_eval, ndim), dtype=dtype, device=device)

    xind = torch.randint(0, nsamples, (batch_size,), device=device)
    dist_batch = torch.permute(dist[:, xind, :], (1, 0, 2))

    interp.fit(nodes_fit, dist_batch)
    x_interp, dx_interp = interp.predict(nodes_eval)

    xtrain[:, ndim] = nodes_eval.reshape(batch_size * n_eval)
    mut = x_interp.reshape(batch_size * n_eval, ndim)
    xtrain[:, :ndim] = mut + sigma * torch.randn_like(mut)
    ytrain[:, :ndim] = dx_interp.reshape(batch_size * n_eval, ndim)
    return xtrain, ytrain


def PFM_(
    dist,
    times,
    net,
    interp=None,
    growth_model=None,
    precomputed_traj=None,
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
    """Train drift and optional growth models with weighted flow matching.

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
    precomputed_traj : torch.Tensor or None, default=None
        Optional precomputed OT trajectories with shape
        ``(n_snaps, n_samples, ndim)``. If ``None``, trajectories are
        computed internally with :func:`pairwise_OT`.
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
        Learning rate for Adam optimizer.
    alpha_ann : float, default=0.5
        Exponential averaging factor for adaptive mass-balance weighting.
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
    loss_hist : list of float
        Optimization loss history.
    """
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

    if precomputed_traj is None:
        batch_ot_samples, _ = pairwise_OT(dist, nb=nb, reg=reg, reg_m=None, method=ot)
    else:
        batch_ot_samples = precomputed_traj
    nsnaps, nsamples, ndim = batch_ot_samples.shape

    mass_vec = torch.ones((nsnaps,), dtype=dist_tensor.dtype, device=device)
    for k in range(1, nsnaps):
        mass_vec[k] = dist[k].shape[0] / dist[0].shape[0]

    uniform_kind = torch.linspace(
        times[0], times[-1], fac * times.shape[0], dtype=dist_tensor.dtype, device=device
    )
    data_nodes = times[None, :].repeat(nsamples, 1).contiguous()
    uniform_nodes = uniform_kind[None, :].repeat(nsamples, 1).contiguous()

    batch_size = nsamples if bs is None else bs
    uniform_batch = uniform_kind[None, :].repeat(batch_size, 1).contiguous()
    data_batch = times[None, :].repeat(batch_size, 1).contiguous()

    batch_ot_samples_ = torch.permute(batch_ot_samples, (1, 0, 2))
    if isinstance(interp, ChebyshevInterpolant):
        if interp.reg is None:
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

    params = [{"params": drift_net.parameters(), "lr": lr}]
    if not zero_growth:
        params.append({"params": growth_model.parameters(), "lr": lr})

    optimizer = torch.optim.Adam(params)
    scheduler_kwargs = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
    default_sched = {"milestones": [1000, 1500, 5000], "gamma": 0.1}
    default_sched.update(scheduler_kwargs)
    scheduler_ = torch.optim.lr_scheduler.MultiStepLR(optimizer, **default_sched)

    dt = uniform_kind[1] - uniform_kind[0]
    lamb = 1.0
    loss_hist = []

    pbar = tqdm(range(n_epochs), desc="pfm", dynamic_ncols=True, disable=not verbose)
    for epoch in pbar:
        optimizer.zero_grad()
        xtrain, ytrain = compute_conditional_distributions(
            interp, batch_ot_samples, batch_size, data_batch, uniform_batch, device=device, sigma=0.001
        )

        if zero_growth:
            growth_eval = torch.zeros((batch_size, fac * nsnaps), dtype=xtrain.dtype, device=device)
        else:
            growth_eval = growth_model(xtrain[:, :ndim]).reshape(batch_size, fac * nsnaps)

        log_mass = torch.zeros_like(growth_eval)
        log_mass[:, 0] = np.log(1.0 / batch_size)
        for k in range(1, fac * nsnaps):
            log_mass[:, k] = log_mass[:, k - 1] + 0.5 * dt * (growth_eval[:, k] + growth_eval[:, k - 1])

        mass_traj = torch.exp(log_mass)
        pred_mass = torch.sum(mass_traj, axis=0)
        weights_ = mass_traj / torch.sum(mass_traj, axis=0, keepdim=True)
        drift_pred = drift_net(xtrain)
        res = (drift_pred - ytrain).pow(2).sum(dim=1).reshape(batch_size, fac * nsnaps)
        if zero_growth:
            weights_ = torch.ones_like(res)

        wcfm_obj = torch.mean(weights_ * res, axis=0)
        mass_balance = (pred_mass[::fac] - mass_vec) ** 2

        l1 = torch.sum(wcfm_obj)
        l2 = torch.sum(mass_balance)

        if not zero_growth and epoch % 10 == 0:
            with torch.no_grad():
                std_l1 = loss_grad_std(l1, drift_net, device)
                std_l2 = loss_grad_std(l2, growth_model, device)
                lamb_hat = std_l1 / std_l2
                lamb = (1 - alpha_ann) * lamb + alpha_ann * lamb_hat

        total_loss = l1
        if not zero_growth:
            total_loss = l1 + lamb * l2
        
        loss_hist.append(total_loss.item())

        total_loss.backward()
        optimizer.step()
        scheduler_.step()

        if verbose:
            pbar.set_postfix(loss=f"{total_loss.item():.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return drift_net, growth_model, loss_hist


__all__ = ["PFM_", "compute_conditional_distributions"]
