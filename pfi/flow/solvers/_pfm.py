"""Flow-matching PFM solver."""

import numpy as np
import torch
from tqdm import tqdm

from ...utils.nns import loss_grad_std
from ..couplings import pairwise_OT
from ..interpolants import ChebyshevInterpolant, select_best_lambda

def compute_conditional_distributions(
    interp,
    trajs,
    batch_size,
    nodes_fit,
    nodes_eval,
    device="cpu",
    sigma=0.001,
    generator=None,
):
    """Build conditional training pairs for drift regression.

    Parameters
    ----------
    interp : object
        Interpolant implementing ``fit`` and ``predict``.
    trajs : torch.Tensor of shape (ntraj, n_snaps, ndim)
        OT-stitched trajectories in trajectory-first layout.
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
    generator : torch.Generator or None, default=None
        Optional RNG for reproducible trajectory sampling and jitter.

    Returns
    -------
    xtrain : torch.Tensor of shape (batch_size * n_eval, ndim + 1)
        Interpolated states with time appended.
    ytrain : torch.Tensor of shape (batch_size * n_eval, ndim)
        Time derivatives of interpolated states.
    """
    ntraj = trajs.shape[0]
    n_eval = nodes_eval.shape[1]
    ndim = trajs.shape[2]
    dtype = trajs.dtype

    xtrain = torch.zeros((batch_size * n_eval, ndim + 1), dtype=dtype, device=device)
    ytrain = torch.zeros((batch_size * n_eval, ndim), dtype=dtype, device=device)

    traj_idx = torch.randint(0, ntraj, (batch_size,), device=device, generator=generator)
    traj_batch = trajs[traj_idx]  # (batch_size, n_snaps, ndim)

    interp.fit(nodes_fit, traj_batch)
    x_interp, dx_interp = interp.predict(nodes_eval)

    xtrain[:, ndim] = nodes_eval.reshape(batch_size * n_eval)
    mut = x_interp.reshape(batch_size * n_eval, ndim)
    xtrain[:, :ndim] = mut + sigma * torch.randn(mut.shape, dtype=mut.dtype, device=mut.device, generator=generator)
    ytrain[:, :ndim] = dx_interp.reshape(batch_size * n_eval, ndim)
    return xtrain, ytrain

def PFM_(
    dist,
    times,
    net,
    interp=None,
    growth_model=None,
    precomputed_traj=None,
    bs=512,
    fac=4,
    nb=1,
    n_epochs=2000,
    lr=1e-3,
    alpha_ann=0.5,
    scheduler_factory=None,
    ot="exact",
    reg=0.1,
    verbose=True,
    generator=None,
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
        ``ChebyshevInterpolant()`` is used.
    growth_model : torch.nn.Module or None, default=None
        Optional growth model receiving spatial coordinates only.
    precomputed_traj : torch.Tensor or None, default=None
        Optional precomputed OT trajectories with shape
        ``(ntraj, n_snaps, ndim)``. If ``None``, trajectories are
        computed internally with ``pairwise_OT``.
    bs : int or None, default=512
        Number of sampled trajectories per optimization step. If ``None``,
        use all sampled trajectories.
    fac : int, default=4
        Number of interpolation nodes per snapshot interval multiplier.
    nb : int, default=1
        Number of OT minibatches.
    n_epochs : int, default=2000
        Number of optimizer iterations.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    alpha_ann : float, default=0.5
        Exponential averaging factor for adaptive mass-balance weighting.
    scheduler_factory : callable or None, default=None
        Callable ``(optimizer) -> scheduler``. If ``None``, uses
        ``MultiStepLR`` with ``milestones=[1000, 1500, 5000], gamma=0.1``.
    ot : str, default="exact"
        Pairwise OT backend name.
    reg : float, default=0.1
        Entropic regularization for OT solver.
    verbose : bool, default=True
        If ``True``, display progress bars.
    generator : torch.Generator or None, default=None
        Optional RNG for reproducible OT sampling and mini-batching.

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
        interp = ChebyshevInterpolant()
    device = dist[0].device
    dtype = dist[0].dtype

    if precomputed_traj is None:
        ot_trajs, _ = pairwise_OT(dist, nb=nb, reg=reg, reg_m=None, method=ot, generator=generator)
    else:
        ot_trajs = precomputed_traj
    nsamples, nsnaps, ndim = ot_trajs.shape  # trajectory-first: (ntraj, nsnaps, ndim)

    mass_vec = torch.ones((nsnaps,), dtype=dtype, device=device)
    for k in range(1, nsnaps):
        mass_vec[k] = dist[k].shape[0] / dist[0].shape[0]

    uniform_kind = torch.linspace(
        times[0], times[-1], fac * times.shape[0], dtype=dtype, device=device
    )
    if bs is None:
        bs = nsamples
    batch_size = min(int(bs), nsamples)

    uniform_batch = uniform_kind[None, :].repeat(batch_size, 1).contiguous()
    data_batch = times[None, :].repeat(batch_size, 1).contiguous()

    if isinstance(interp, ChebyshevInterpolant):
        if interp.reg is None:
            data_nodes = times[None, :].repeat(nsamples, 1).contiguous()
            uniform_nodes = uniform_kind[None, :].repeat(nsamples, 1).contiguous()
            best_lam, _, _ = select_best_lambda(
                ot_trajs,
                data_nodes,
                uniform_nodes,
                verbose=verbose,
            )
            interp.reg = best_lam

    drift_net = net
    zero_growth = growth_model is None

    params = [{"params": drift_net.parameters(), "lr": lr}]
    if not zero_growth:
        params.append({"params": growth_model.parameters(), "lr": lr})

    optimizer = torch.optim.Adam(params)
    if scheduler_factory is None:
        def scheduler_factory(opt):
            return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000, 1500, 5000], gamma=0.1)
    scheduler = scheduler_factory(optimizer)

    dt = uniform_kind[1] - uniform_kind[0]
    lam = 1.0
    loss_hist = []

    pbar = tqdm(range(n_epochs), desc="pfm", dynamic_ncols=True, disable=not verbose)
    for epoch in pbar:
        optimizer.zero_grad()
        xtrain, ytrain = compute_conditional_distributions(
            interp, ot_trajs, batch_size, data_batch, uniform_batch, device=device, sigma=0.001, generator=generator,
        )

        drift_pred = drift_net(xtrain)
        res = (drift_pred - ytrain).pow(2).sum(dim=1).reshape(batch_size, fac * nsnaps)

        if zero_growth:
            # Mass-weighted CFM objective (uniform weights when no growth)
            weights = torch.ones_like(res)
            loss_drift = torch.sum(torch.mean(weights * res, axis=0))
        else:
            growth_eval = growth_model(xtrain).reshape(batch_size, fac * nsnaps)
            # Trapezoidal integration: d/dt log(mass) = g(t)
            log_mass = torch.zeros_like(growth_eval)
            log_mass[:, 0] = np.log(1.0 / batch_size)
            for k in range(1, fac * nsnaps):
                log_mass[:, k] = log_mass[:, k - 1] + 0.5 * dt * (growth_eval[:, k] + growth_eval[:, k - 1])
            mass_traj = torch.exp(log_mass)
            pred_mass = torch.sum(mass_traj, axis=0)
            # Mass-weighted CFM objective
            weights = mass_traj / torch.sum(mass_traj, axis=0, keepdim=True)
            wcfm_obj = torch.mean(weights * res, axis=0)
            mass_balance = (pred_mass[::fac] - mass_vec) ** 2
            loss_drift = torch.sum(wcfm_obj)
            loss_mass = torch.sum(mass_balance)

            # Rebalance drift/mass losses by gradient std ratio
            if epoch % 10 == 0:
                with torch.no_grad():
                    std_l1 = loss_grad_std(loss_drift, drift_net, device)
                    std_l2 = loss_grad_std(loss_mass, growth_model, device)
                    lam_hat = std_l1 / std_l2
                    lam = (1 - alpha_ann) * lam + alpha_ann * lam_hat

        if zero_growth:
            total_loss = loss_drift
        else:
            total_loss = loss_drift + lam * loss_mass

        loss_hist.append(total_loss.item())

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose:
            pbar.set_postfix(loss=f"{total_loss.item():.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return drift_net, growth_model, loss_hist

__all__ = ["PFM_", "compute_conditional_distributions"]
