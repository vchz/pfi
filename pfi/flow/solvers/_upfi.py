"""Neural-ODE-style UPFI solver."""

import torch
from tqdm import tqdm

def _sample_minibatch(
    dist,
    bs,
    generator=None,
):
    """Sample an independent mini-batch at each snapshot.

    Parameters
    ----------
    dist : list of torch.Tensor, length nsnaps
        ``dist[k]`` has shape ``(n_k, ndim)`` on a common device.
    bs : int
        Number of samples drawn per snapshot.

    Returns
    -------
    batch : torch.Tensor of shape (nsnaps, bs, ndim)
        Independent mini-batches sampled at each snapshot.
    """
    nsnaps = len(dist)
    device = dist[0].device
    dtype = dist[0].dtype

    batch = torch.zeros((nsnaps, bs, dist[0].shape[1]), dtype=dtype, device=device)
    for k in range(nsnaps):
        idx = torch.randint(0, dist[k].shape[0], (bs,), device=dist[k].device, generator=generator)
        batch[k] = dist[k].index_select(0, idx)
    return batch

def _custom_ode_int(
    flow_model,
    growth_model,
    x0,
    t0,
    t1,
    n_steps,
    logw0,
    alpha_wfr,
):
    """Integrate one snapshot interval with explicit Euler steps.

    Parameters
    ----------
    flow_model : torch.nn.Module
        Flow model returning deterministic drift for ``stoch=False``.
    growth_model : torch.nn.Module or None
        Optional growth model returning a scalar rate.
    x0 : torch.Tensor of shape (batch_size, ndim)
        Initial states.
    t0 : float or torch.Tensor
        Initial time.
    t1 : float or torch.Tensor
        Final time.
    n_steps : int
        Number of Euler steps.
    logw0 : torch.Tensor of shape (batch_size,)
        Initial log-masses.
    alpha_wfr : float
        Relative weight of growth in the regularization term.

    Returns
    -------
    x : torch.Tensor of shape (batch_size, ndim)
        Integrated final states.
    log_mass_1 : torch.Tensor of shape (batch_size,)
        Integrated final log-masses.
    reg : torch.Tensor of shape ()
        Accumulated regularization contribution over the interval.
    """
    batch_size = x0.shape[0]
    delta_t = t1 - t0
    n_steps = max(1, int(n_steps))
    h = delta_t / n_steps

    x = x0
    log_mass = logw0
    t = torch.full((batch_size,), t0, dtype=x0.dtype, device=x0.device)
    reg = torch.zeros((batch_size,), dtype=x0.dtype, device=x0.device)
    for _ in range(n_steps):
        inp = torch.cat([x, t[:, None]], dim=1)
        drift = flow_model(inp, stoch=False)
        if growth_model is None:
            g = torch.zeros((batch_size,), dtype=x0.dtype, device=x0.device)
        else:
            g = growth_model(inp).squeeze(-1)

        x = x + h * drift
        log_mass = log_mass + h * g
        # WFR regularization: kinetic energy + growth rate, weighted by current mass
        reg = reg + h * (drift.pow(2).sum(dim=1) + alpha_wfr * g.pow(2)) * torch.exp(log_mass)
        t = t + h

    # Scale by Δt so total reg is independent of n_steps
    return x, log_mass, reg.sum() * (t1 - t0)

def UPFI_(
    dist,
    times,
    net,
    growth_model=None,
    n_epochs=2000,
    lr=1e-3,
    bs=512,
    n_steps=2,
    reg_wfr=1,
    reg=0.1,
    alpha_wfr=1.0,
    reach=10.0,
    scheduler_factory=None,
    verbose=True,
    generator=None,
):
    """Fit flow and growth jointly with teacher-forced ODE matching.

    Parameters
    ----------
    dist : list of torch.Tensor
        Snapshot list where ``dist[k]`` has shape ``(n_k, ndim)``.
    times : torch.Tensor of shape (n_snaps,)
        Snapshot times.
    net : torch.nn.Module
        Flow model returning probability-flow drift for ``stoch=False``.
    growth_model : torch.nn.Module or None, default=None
        Growth model mapping spatial states to scalar growth.
    n_epochs : int, default=2000
        Number of optimizer iterations.
    lr : float, default=1e-3
        Learning rate.
    bs : int, default=512
        Number of sampled trajectories per optimization step.
    n_steps : int, default=2
        Number of integration steps between two snapshots.
    reg_wfr : float, default=1
        Coefficient for WFR-style regularization.
    reg : float, default=0.1
        Coefficient for entropic regularization in the loss.
    alpha_wfr : float, default=1.0
        Relative weight of growth contribution in WFR-style regularization.
    reach : float, default=10.0
        Reach parameter passed to ``geomloss.SamplesLoss``.
    scheduler_factory : callable or None, default=None
        Callable ``(optimizer) -> scheduler``. If ``None``, uses
        ``MultiStepLR`` with ``milestones=[1000, 1500, 8000, 15000], gamma=0.1``.
    verbose : bool, default=True
        If ``True``, display progress bar.
    generator : torch.Generator or None, default=None
        Optional RNG for reproducible sampling.

    Returns
    -------
    drift_net : torch.nn.Module
        Trained flow model.
    growth_model : torch.nn.Module or None
        Trained growth model.
    loss_hist : list of float
        Optimization loss history.
    """
    import geomloss

    device = dist[0].device
    dtype = dist[0].dtype
    nsnaps = len(dist)

    nsamples = min(s.shape[0] for s in dist)

    if bs is None:
        bs = nsamples
    bs = min(int(bs), nsamples)

    mass_vec = torch.tensor([d.shape[0] / dist[0].shape[0] for d in dist], dtype=dtype, device=device)

    drift_net = net

    if growth_model is None:
        reach = None
        reg_wfr = 0

    params = [{"params": drift_net.parameters(), "lr": lr}]
    if growth_model is not None:
        params.append({"params": growth_model.parameters(), "lr": lr})
    optimizer = torch.optim.Adam(params)

    if scheduler_factory is None:
        def scheduler_factory(opt):
            return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000, 1500, 8000, 15000], gamma=0.1)
    scheduler = scheduler_factory(optimizer)

    loss_fn = geomloss.SamplesLoss(loss="sinkhorn", reach=reach, blur=reg)

    loss_hist = []
    pbar = tqdm(range(n_epochs), desc="upfi", dynamic_ncols=True, disable=not verbose)
    for _ in pbar:
        optimizer.zero_grad()

        x_batch_t = _sample_minibatch(dist, bs=bs, generator=generator)

        x_hat = []
        logw_hat = []
        reg_loss = torch.zeros((), dtype=dtype, device=device)
        for k in range(nsnaps - 1):
            x0 = x_batch_t[k]
            # Log-mass initialized to log(relative snapshot size / batch_size)
            log_mass_0 = torch.log(mass_vec[k] / bs) * torch.ones((bs,), dtype=dtype, device=device)
            x1, log_mass_1, reg_interval = _custom_ode_int(
                drift_net,
                growth_model,
                x0,
                times[k],
                times[k + 1],
                n_steps,
                log_mass_0,
                alpha_wfr,
            )
            x_hat.append(x1)
            logw_hat.append(log_mass_1)
            reg_loss = reg_loss + reg_interval

        x_hat = torch.stack(x_hat, dim=0)
        logw_hat = torch.stack(logw_hat, dim=0)
        x_obs = torch.stack([x_batch_t[k] for k in range(1, nsnaps)], dim=0)
        logw_obs = torch.stack(
            [torch.log(mass_vec[k] / bs) * torch.ones((bs,), dtype=dtype, device=device) for k in range(1, nsnaps)],
            dim=0,
        )

        fit_vec = loss_fn(
            torch.exp(logw_hat),
            x_hat,
            torch.exp(logw_obs),
            x_obs,
        )
        fit_loss = fit_vec.sum()

        total_loss = fit_loss + reg_wfr * reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(float(total_loss.item()))
        if verbose:
            pbar.set_postfix(
                total=f"{total_loss.item():.3e}",
                fit=f"{fit_loss.item():.3e}",
                wfr=f"{reg_loss.item():.3e}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    return drift_net, growth_model, loss_hist

__all__ = ["UPFI_"]
