"""Denoising score-matching (DSM) solvers and sampling utilities."""

import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from ...utils.nns import FastTensorDataLoader, loss_grad_std

def _generate_noisy_training_data_batch(
    Dist,
    tp,
    L,
    generator=None,
):
    """Generate noisy DSM training inputs in samples-first layout.

    Parameters
    ----------
    Dist : torch.Tensor of shape (nsnaps, nsamples, ndim)
        Snapshot samples on target device.
    tp : torch.Tensor of shape (nsnaps,)
        Snapshot times on target device.
    L : int
        Number of noise levels.

    Returns
    -------
    x_train : torch.Tensor of shape (nsamples, nsnaps, L, ndim + 2)
        Noisy training inputs with appended noise level and time.
    x_data : torch.Tensor of shape (nsamples, nsnaps, ndim)
        Clean data tensor.
    sigma : torch.Tensor of shape (L,)
        Noise schedule.
    """
    from ...utils.data import geometric_sequence

    nsnaps, nsamples, ndim = Dist.shape
    device = Dist.device
    dtype = Dist.dtype
    sigma = torch.as_tensor(geometric_sequence(L), dtype=dtype, device=device)

    Dist = Dist.permute(1, 0, 2)  # (nsamples, nsnaps, ndim)
    noise = torch.randn((nsamples, nsnaps, L, ndim), dtype=dtype, device=device, generator=generator)
    x_noisy = Dist[:, :, None, :] + sigma[None, None, :, None] * noise  # (nsamples, nsnaps, L, ndim)

    x_train = torch.zeros((nsamples, nsnaps, L, ndim + 2), dtype=dtype, device=device)
    x_train[:, :, :, :ndim] = x_noisy
    x_train[:, :, :, ndim] = sigma[None, None, :]       # noise level per L slot
    x_train[:, :, :, ndim + 1] = tp[None, :, None]      # time per snapshot
    x_data = Dist  # (nsamples, nsnaps, ndim)

    return x_train, x_data, sigma

def DSM_(
    dist,
    times,
    net,
    L=5,
    n_epochs=5000,
    bs=None,
    adaptive_weights=0,
    lr=5e-4,
    scheduler_factory=None,
    verbose=True,
    generator=None,
):
    """Train a score network with denoising score matching.

    Parameters
    ----------
    dist : list of torch.Tensor, length nsnaps
        ``dist[k]`` has shape ``(n_k, ndim)`` and must already be
        ``float32`` on the same device as ``net``.
    times : torch.Tensor of shape (nsnaps,)
        Snapshot times as ``float32`` on the same device as ``net``.
    net : torch.nn.Module
        Score network that accepts inputs with last dimension ``ndim + 2``.
    L : int, default=5
        Number of noise levels.
    n_epochs : int, default=5000
        Number of optimization epochs.
    bs : int or None, default=None
        Mini-batch size over sample dimension. If ``None``, uses all samples.
    adaptive_weights : int, default=0
        If set to ``1``, enable adaptive per-time weighting.
    lr : float, default=5e-4
        Learning rate.
    scheduler_factory : callable or None, default=None
        Callable ``(optimizer) -> scheduler``. If ``None``, uses
        ``MultiStepLR`` with ``milestones=[2500, 6500, 8500], gamma=0.1``.
    verbose : bool, default=True
        If ``True``, show progress bars and diagnostics.

    Returns
    -------
    net : torch.nn.Module
        Trained score network.
    loss_hist : list of float
        Epoch-wise training loss values.
    sigma : torch.Tensor of shape (L,)
        Noise schedule used during training.
    """
    device = dist[0].device
    nsamples = min(s.shape[0] for s in dist)
    dist = torch.stack([s[torch.randperm(s.shape[0], device=device, generator=generator)[:nsamples]] for s in dist], dim=0)
    dtype = dist.dtype

    nsnaps, nsamples, ndim = dist.shape
    if bs is None:
        bs = nsamples
    bs = min(int(bs), nsamples)

    x_train, x_data, sigma = _generate_noisy_training_data_batch(
        dist,
        times,
        L,
        generator=generator,
    )

    # init_model must be called here: the solver generates the noise schedule
    net.init_model(x_train, generator=generator)

    loader = FastTensorDataLoader(
        x_train,
        x_data,
        batch_size=bs,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.Adam(list(net.parameters()), lr=lr)
    if scheduler_factory is None:
        def scheduler_factory(opt):
            return MultiStepLR(opt, milestones=[2500, 6500, 8500], gamma=0.1)
    scheduler = scheduler_factory(optimizer)

    time_weights = torch.ones((nsnaps,), dtype=dtype, device=device)
    alpha_ann = 0.5
    weight_update_interval = 10
    weight_decay = 1e-4

    sigma_reshaped = sigma.view(1, L, 1)  # broadcasts with (bs, L, ndim)
    pbar = tqdm(range(n_epochs), desc="DSM", dynamic_ncols=True, disable=not verbose)
    loss_hist = []

    for epoch in pbar:
        for xbatch, ybatch in loader:
            # xbatch: (bs, nsnaps, L, ndim+2)   ybatch: (bs, nsnaps, ndim)
            optimizer.zero_grad()

            uhat = net(xbatch)
            lcomp = torch.zeros((nsnaps,), dtype=dtype, device=device)
            std_ = torch.zeros((nsnaps,), dtype=dtype, device=device)

            for tind in range(nsnaps):
                u_pred = uhat[:, tind, :, :]
                x_t = xbatch[:, tind, :, :ndim]
                x_ref = ybatch[:, tind, :ndim][:, None, :]

                # DSM target: optimal score is -(x_noisy - x_clean) / σ²
                u_true = (x_t - x_ref) / (sigma_reshaped ** 2)
                residual = u_pred + u_true
                residual_squared = residual.pow(2).mean(dim=(0, 2))  # mean over (bs, ndim) per L level
                loss_sum = 0.5 * torch.sum((sigma ** 2) * residual_squared)

                if adaptive_weights == 1 and epoch % weight_update_interval == 0:
                    with torch.no_grad():
                        std_[tind] = loss_grad_std(loss_sum, net, device)

                lcomp[tind] = loss_sum

            # EMA toward inverse gradient-std weighting across snapshots
            if adaptive_weights == 1 and epoch % weight_update_interval == 0:
                with torch.no_grad():
                    lamb_hat = torch.max(std_) / std_
                    time_weights = (1 - alpha_ann) * time_weights + alpha_ann * lamb_hat
                    time_weights = time_weights / torch.sum(time_weights)

            loss = sum(time_weights[tind] * lcomp[tind] for tind in range(nsnaps))
            # Manual L2 regularization (not handled by Adam by default)
            weight_norm = sum((p ** 2).sum() for p in net.parameters())
            loss = loss + weight_decay * weight_norm

            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_hist.append(loss.item())
        if verbose:
            pbar.set_postfix(loss=f"{loss.item():.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return net, loss_hist, sigma

def generate_data_DSM(
    net,
    x0,
    times,
    sigma,
    maxiter=100,
    generator=None,
):
    """Sample from a trained DSM model using annealed Langevin dynamics.

    Parameters
    ----------
    net : torch.nn.Module
        Trained score model.
    x0 : torch.Tensor of shape (nsamples, ndim + 2)
        Initial states including noise-level and time columns.
    times : torch.Tensor of shape (nsamples,)
        Time values assigned to generated samples.
    sigma : torch.Tensor of shape (L,)
        Noise schedule, typically ``ScoreEstimator.sigma_``.
    maxiter : int, default=100
        Number of Langevin steps per noise level.

    Returns
    -------
    sol : torch tensor of shape (nsamples, ndim + 2)
        Generated states including auxiliary columns.
    """
    eps = 1e-4
    sol = x0.clone()
    nsamples = sol.shape[0]
    ndim = sol.shape[1] - 2
    sol[:, ndim + 1] = times

    sigma = sigma.to(dtype=sol.dtype, device=sol.device)
    stop_k = int(torch.argmin(torch.abs(sigma - sigma[-1])).item())

    for k in range(stop_k + 1):
        alpha = eps * ((sigma[k] ** 2) / (sigma[-1] ** 2))
        sol[:, ndim] = sigma[k]

        for _ in range(maxiter):
            z = torch.randn((nsamples, ndim), dtype=sol.dtype, device=sol.device, generator=generator)
            score = net(sol)
            sol[:, 0:ndim] = sol[:, 0:ndim] + 0.5 * alpha * score + torch.sqrt(alpha) * z

    sol = torch.reshape(sol, (nsamples, ndim + 2))

    return sol
