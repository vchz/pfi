"""Neural-ODE-style UPFI solver."""

import torch
from tqdm import tqdm


def sample_minibatch(
    dist,
    bs,
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
        idx = torch.randint(0, dist[k].shape[0], (bs,), device=dist[k].device)
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
    """Integrate one interval with Euler in teacher-forcing mode."""
    batch_size = x0.shape[0]
    delta_t = t1 - t0
    n_steps = max(1, int(n_steps))
    h = delta_t / n_steps

    x = x0
    logw = logw0
    t = torch.full((batch_size,), t0, dtype=x0.dtype, device=x0.device)
    reg = torch.zeros((batch_size,), dtype=x0.dtype, device=x0.device)
    for _ in range(n_steps):
        inp = torch.cat([x, t[:, None]], dim=1)
        drift = flow_model(inp, stoch=False)
        if growth_model is None:
            g = torch.zeros((batch_size,), dtype=x0.dtype, device=x0.device)
        else:
            g = growth_model(x).squeeze(-1)

        x = x + h * drift
        logw = logw + h * g
        reg = reg + h * (drift.pow(2).sum(dim=1) + alpha_wfr * g.pow(2)) * torch.exp(logw)
        t = t + h

    return x, logw, reg.sum()*(t1 - t0)


def UPFI_(
    dist,
    times,
    net,
    growth_model=None,
    n_epochs=2000,
    lr=1e-3,
    bs=None,
    n_steps=2,
    reg_wfr=1,
    reg=0.1,
    alpha_wfr=1.0,
    reach=10.0,
    scheduler_kwargs=None,
    verbose=True,
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
    n_steps : int, default=20
        Number of integration steps between two snapshots.
    reg_wfr : float, default=0.1
        Coefficient for WFR-style regularization.
    reg : float, default=0.1
        Coefficient for entropic regularization in the loss.
    alpha_wfr : float, default=1.0
        Relative weight of growth contribution in WFR-style regularization.
    reach : float, default=5.0
        Reach parameter passed to ``geomloss.SamplesLoss``.
    scheduler_kwargs : dict or None, default=None
        Optional kwargs for ``MultiStepLR``.
    verbose : bool, default=True
        If ``True``, display progress bar.

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
    dist_tensor = torch.stack(
        [s[torch.randperm(s.shape[0], device=s.device)[:nsamples]] for s in dist],
        dim=0,
    )
    x_mean = dist_tensor.mean(dim=(0, 1))
    x_std = dist_tensor.std(dim=(0, 1))
    t_mean = times.mean().to(device=device, dtype=x_mean.dtype)
    t_std = times.std().to(device=device, dtype=x_std.dtype)
    net.set_scales(torch.cat([x_mean, t_mean[None]]), torch.cat([x_std, t_std[None]]))
    if growth_model is not None and hasattr(growth_model, "set_scales"):
        growth_model.set_scales(x_mean, x_std)

    if bs is None:
        bs = max(1, nsamples // 4)
    mass_vec = torch.tensor([d.shape[0] / dist[0].shape[0] for d in dist], dtype=dtype, device=device)

    drift_net = net.to(device)
    if growth_model is not None:
        growth_model = growth_model.to(device)

    if growth_model is None:
        reach = None
        reg_wfr = 0

    params = [{"params": drift_net.parameters(), "lr": lr}]
    if growth_model is not None:
        params.append({"params": growth_model.parameters(), "lr": lr})
    optimizer = torch.optim.Adam(params)

    scheduler_kwargs = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
    default_sched = {"milestones": [1000, 1500, 8000, 15000], "gamma": 0.1}
    default_sched.update(scheduler_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **default_sched)

    loss_fn = geomloss.SamplesLoss(loss="sinkhorn", reach=reach, blur=reg)

    loss_hist = []
    pbar = tqdm(range(n_epochs), desc="upfi", dynamic_ncols=True, disable=not verbose)
    for _ in pbar:
        optimizer.zero_grad()

        x_batch_t = sample_minibatch(dist, bs=bs)
        x_batch = [x_batch_t[k] for k in range(nsnaps)]

        x_hat = []
        logw_hat = []
        reg_loss = torch.zeros((), dtype=dtype, device=device)
        for k in range(nsnaps - 1):
            x0 = x_batch[k]
            lw0 = torch.log(mass_vec[k] / bs) * torch.ones((bs,), dtype=dtype, device=device)
            x1, lw1, reg_k = _custom_ode_int(
                drift_net,
                growth_model,
                x0,
                times[k],
                times[k + 1],
                n_steps,
                lw0,
                alpha_wfr,
            )
            x_hat.append(x1)
            logw_hat.append(lw1)
            reg_loss = reg_loss + reg_k

        x_hat = torch.stack(x_hat, dim=0)
        logw_hat = torch.stack(logw_hat, dim=0)
        x_obs = torch.stack(x_batch[1:], dim=0)
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
