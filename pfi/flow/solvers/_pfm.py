"""Flow-matching PFM solver."""

import numpy as np
import torch
from tqdm import tqdm

from ...utils.nns import loss_grad_std
from ..paths import ExactOT
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

def _velocity(net, x, t):
    """Deterministic (probability-flow) velocity ``net([x, t])`` at time ``t``,
    detached. Evaluated with gradients enabled, since some models compute it by
    automatic differentiation (gradient flows, divergence of the diffusion)."""
    with torch.enable_grad():
        t_col = torch.full((x.shape[0], 1), float(t), dtype=x.dtype, device=x.device)
        xt = torch.cat([x.detach(), t_col], dim=1)
        return net(xt).detach()


def _model_push(net, times, n_steps=8):
    """Reference dynamics of a flow model: ``push(x, k)`` integrates the model's
    deterministic (probability-flow) velocity from ``times[k]`` to
    ``times[k + 1]`` with ``n_steps`` Runge-Kutta 4 steps (result detached)."""
    def push(x, k):
        t0, t1 = float(times[k]), float(times[k + 1])
        h = (t1 - t0) / n_steps

        def v(y, t):
            return _velocity(net, y, t)

        x = x.detach()
        for i in range(n_steps):
            t = t0 + i * h
            k1 = v(x, t)
            k2 = v(x + 0.5 * h * k1, t + 0.5 * h)
            k3 = v(x + 0.5 * h * k2, t + 0.5 * h)
            k4 = v(x + h * k3, t + h)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
    return push


def PFM_(
    dist,
    times,
    net,
    interp=None,
    paths=None,
    growth_model=None,
    precomputed_traj=None,
    bs=512,
    fac=4,
    n_epochs=2000,
    lr=1e-3,
    alpha_ann=0.5,
    scheduler_factory=None,
    n_rounds=1,
    n_epochs_round=None,
    push_steps=8,
    history=None,
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
    paths : pfi.flow.paths.PathEnsemble or None, default=None
        Path ensemble providing the trajectories (``fit(dist, push)`` then
        ``sample()``). If ``None``, ``ExactOT()``.
    growth_model : torch.nn.Module or None, default=None
        Optional growth model receiving spatial coordinates only.
    precomputed_traj : torch.Tensor or None, default=None
        Optional precomputed OT trajectories with shape
        ``(ntraj, n_snaps, ndim)``, used instead of ``paths``. If ``None``,
        trajectories are sampled from ``paths``.
    bs : int or None, default=512
        Number of sampled trajectories per optimization step. If ``None``,
        use all sampled trajectories.
    fac : int, default=4
        Number of interpolation nodes per snapshot interval multiplier.
    n_epochs : int, default=2000
        Number of optimizer iterations.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    alpha_ann : float, default=0.5
        Exponential averaging factor for adaptive mass-balance weighting.
    scheduler_factory : callable or None, default=None
        Callable ``(optimizer) -> scheduler``. If ``None``, uses
        ``MultiStepLR`` with ``milestones=[1000, 1500, 5000], gamma=0.1``.
    n_rounds : int, default=1
        Number of rounds of the iterated coupling. Round 1 builds the paths
        without a model; every further round builds them again with the
        current model as reference dynamics (each interval's source pushed by
        the model's flow, see ``pfi.flow.paths``) and retrains the model,
        warm-started, for ``n_epochs`` epochs with a fresh optimizer and
        scheduler. ``1``: no iteration (original behaviour).
    n_epochs_round : int or None, default=None
        Optimizer iterations of every round after the first; ``None``: ``n_epochs``.
    push_steps : int, default=8
        Runge-Kutta 4 steps per snapshot interval when pushing with the model.
    history : list or None, default=None
        If a list, receives one dict per round: ``round``, ``change`` (relative
        change of the model's velocity on the round's path nodes during the
        round's training, at most 5000 nodes) and ``loss`` (last loss).
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

    if precomputed_traj is not None and (paths is not None or n_rounds > 1):
        raise ValueError("precomputed_traj cannot be combined with paths or n_rounds > 1.")
    if paths is None and precomputed_traj is None:
        paths = ExactOT()

    loss_hist = []
    for rnd in range(n_rounds):
        push = None if rnd == 0 else _model_push(net, times, push_steps)
        if precomputed_traj is None:
            paths.fit(dist, push=push, generator=generator)
            ot_trajs, _ = paths.sample(generator=generator)
        else:
            ot_trajs = precomputed_traj
        if verbose and n_rounds > 1:
            print(f"pfm: round {rnd + 1}/{n_rounds}", flush=True)
        if history is not None:
            n_nodes = ot_trajs.shape[0] * ot_trajs.shape[1]
            sel = torch.randperm(n_nodes, device=ot_trajs.device, generator=generator)[:5000]
            nodes = ot_trajs.reshape(n_nodes, -1)[sel]
            node_t = times.repeat(ot_trajs.shape[0])[sel]
            v_before = torch.cat([_velocity(net, nodes[node_t == t], t) for t in times])
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
                return torch.optim.lr_scheduler.MultiStepLR(
                    opt,
                    milestones=[1000, 1500, 5000],
                    gamma=0.1,
                )
        scheduler = scheduler_factory(optimizer)

        dt = uniform_kind[1] - uniform_kind[0]
        lam = 1.0

        n_ep = n_epochs if rnd == 0 or n_epochs_round is None else n_epochs_round
        pbar = tqdm(range(n_ep), desc="pfm", dynamic_ncols=True, disable=not verbose)
        for epoch in pbar:
            optimizer.zero_grad()
            xtrain, ytrain = compute_conditional_distributions(
                interp,
                ot_trajs,
                batch_size,
                data_batch,
                uniform_batch,
                device=device,
                sigma=0.001,
                generator=generator,
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
                    log_mass[:, k] = (
                        log_mass[:, k - 1] + 0.5 * dt * (growth_eval[:, k] + growth_eval[:, k - 1])
                    )
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
                pbar.set_postfix(
                    loss=f"{total_loss.item():.3e}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        if history is not None:
            v_after = torch.cat([_velocity(net, nodes[node_t == t], t) for t in times])
            change = float((v_after - v_before).norm() / v_after.norm().clamp_min(1e-30))
            history.append({"round": rnd + 1, "change": change, "loss": loss_hist[-1]})
            if verbose:
                print(
                    f"pfm: round {rnd + 1}: velocity change {change:.3e}, "
                    f"loss {loss_hist[-1]:.3e}",
                    flush=True,
                )

    return drift_net, growth_model, loss_hist

__all__ = ["PFM_", "compute_conditional_distributions"]
