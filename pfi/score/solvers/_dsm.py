"""Denoising score-matching (DSM) solvers and sampling utilities."""

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from ...utils.nns import FastTensorDataLoader, FreezeVarDNN, loss_grad_std


def geometric_sequence(
    L,
):
    """Build the geometric noise schedule used by DSM.

    Parameters
    ----------
    L : int
        Number of noise levels.

    Returns
    -------
    sigma : ndarray of shape (L,)
        Geometrically decaying standard deviations.
    """
    r = np.exp((-2 / L) * np.log(10))
    sigma = np.zeros((L,))
    for i in range(0, L):
        sigma[i] = 1 * r ** i
    return sigma


def generate_noisy_training_data_batch(
    Dist,
    ndim,
    tp,
    L,
    nsamples,
    nsnaps,
):
    """Generate noisy DSM training inputs and normalization statistics.

    Parameters
    ----------
    Dist : torch.Tensor of shape (nsnaps, nsamples, ndim)
        Snapshot samples on target device.
    ndim : int
        State dimension.
    tp : torch.Tensor of shape (nsnaps,)
        Snapshot times on target device.
    L : int
        Number of noise levels.
    nsamples : int
        Number of samples per snapshot.
    nsnaps : int
        Number of snapshots.
    Returns
    -------
    x_train : torch.Tensor of shape (nsnaps, L, nsamples, ndim + 2)
        Noisy training inputs with appended noise level and time.
    x_data : torch.Tensor of shape (nsnaps, nsamples, ndim)
        Clean data tensor.
    x_mean : torch.Tensor of shape (1, ndim + 2)
        Feature-wise mean used for input normalization.
    x_std : torch.Tensor of shape (1, ndim + 2)
        Feature-wise standard deviation used for input normalization.
    sigma : torch.Tensor of shape (L,)
        Noise schedule.
    """
    device = Dist.device
    dtype = Dist.dtype
    sigma = torch.as_tensor(geometric_sequence(L), dtype=dtype, device=device)

    noise = torch.randn((nsnaps, L, nsamples, ndim), dtype=dtype, device=device)
    x_noisy = Dist[:, None, :, :] + sigma[None, :, None, None] * noise

    x_train = torch.zeros((nsnaps, L, nsamples, ndim + 2), dtype=dtype, device=device)
    x_train[:, :, :, :ndim] = x_noisy
    x_train[:, :, :, ndim] = sigma[None, :, None]
    x_train[:, :, :, ndim + 1] = tp[:, None, None]
    x_data = Dist

    x_mean = x_train.mean(dim=(0, 1, 2))
    x_std = x_train.std(dim=(0, 1, 2))

    return x_train, x_data, x_mean, x_std, sigma


def DSM_(
    dist,
    times,
    net,
    L=10,
    n_epochs=2000,
    bs=None,
    adp_flag=1,
    lr=1e-4,
    scheduler_kwargs=None,
    verbose=True,
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
        Score network that accepts inputs of shape
        ``(nsnaps, L, batch_size, ndim + 2)`` after batching.
    L : int, default=10
        Number of noise levels.
    n_epochs : int, default=2000
        Number of optimization epochs.
    bs : int or None, default=None
        Mini-batch size over sample dimension. If ``None``, uses all samples.
    adp_flag : int, default=0
        If set to ``1``, enable adaptive per-time weighting.
    lr : float, default=1e-4
        Learning rate.
    scheduler_kwargs : dict or None, default=None
        Keyword arguments used to configure
        ``torch.optim.lr_scheduler.MultiStepLR``.
        Defaults to ``milestones=[2500, 6500, 8500], gamma=0.1``.
    verbose : bool, default=True
        If ``True``, show progress bars and diagnostics.

    Returns
    -------
    net : torch.nn.Module
        Trained score network.
    loss_hist : list of float
        Epoch-wise training loss values.
    min_noise : float
        Smallest noise level used by the DSM schedule.
    """
    device = dist[0].device
    nsamples = min(s.shape[0] for s in dist)
    dist = torch.stack([s[torch.randperm(s.shape[0], device=device)[:nsamples]] for s in dist], dim=0)
    dtype = dist.dtype

    nsnaps, nsamples, ndim = dist.shape
    if bs is None:
        bs = nsamples

    x_train, x_data, x_mean, x_std, sigma = generate_noisy_training_data_batch(
        dist,
        ndim,
        times,
        L,
        nsamples,
        nsnaps,
    )

    net.set_scales(x_mean, x_std)

    loader = FastTensorDataLoader(
        torch.permute(x_train, (2, 0, 1, 3)),
        torch.permute(x_data, (1, 0, 2)),
        batch_size=bs,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(list(net.parameters()), lr=lr)
    scheduler_kwargs = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
    default_sched = {"milestones": [2500, 6500, 8500], "gamma": 0.1}
    default_sched.update(scheduler_kwargs)
    scheduler = MultiStepLR(optimizer, **default_sched)

    c_ = torch.ones((nsnaps,), dtype=dtype, device=device)
    alpha_ann = 0.5
    adapt_int = 10
    weight_decay = 1e-4

    sigma_tensor = sigma
    sigma_reshaped = sigma_tensor.view(L, 1, 1)
    pbar = tqdm(range(n_epochs), desc="DSM", dynamic_ncols=True, disable=not verbose)
    loss_hist = []

    for epoch in pbar:
        for _, (xbatch, ybatch) in enumerate(loader):
            optimizer.zero_grad()

            xbatch = torch.permute(xbatch, (1, 2, 0, 3))
            ybatch = torch.permute(ybatch, (1, 0, 2))

            uhat = net(xbatch)
            lcomp = torch.zeros((nsnaps,), dtype=dtype, device=device)
            std_ = torch.zeros((nsnaps,), dtype=dtype, device=device)

            for tind in range(nsnaps):
                u_pred = uhat[tind]
                x_t = xbatch[tind, :, :, :ndim]
                x_ref = ybatch[tind, :, :ndim][None, :, :]

                u_true = (x_t - x_ref) / (sigma_reshaped ** 2)
                residual = u_pred + u_true
                residual_squared = residual.pow(2).mean(dim=(1, 2))
                loss_sum = 0.5 * torch.sum((sigma_tensor ** 2) * residual_squared)

                if adp_flag == 1 and epoch % adapt_int == 0:
                    with torch.no_grad():
                        std_[tind] = loss_grad_std(loss_sum, net, device)

                lcomp[tind] = loss_sum

            if adp_flag == 1 and epoch % adapt_int == 0:
                with torch.no_grad():
                    lamb_hat = torch.max(std_) / std_
                    c_ = (1 - alpha_ann) * c_ + alpha_ann * lamb_hat
                    c_ = c_ / torch.sum(c_)

            loss = sum(c_[tind] * lcomp[tind] for tind in range(nsnaps))
            weight_norm = sum((p ** 2).sum() for p in net.parameters())
            loss = loss + weight_decay * weight_norm

            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_hist.append(loss.item())
        if verbose:
            pbar.set_postfix(loss=f"{loss.item():.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    min_noise = sigma[-1]
    return net, loss_hist, min_noise


def generate_data_DSM(
    maxiter,
    infNet,
    init_,
    time_,
    L,
    noise_lvl=0,
):
    """Sample from a trained DSM model using annealed Langevin dynamics.

    Parameters
    ----------
    maxiter : int
        Number of Langevin steps per noise level.
    infNet : torch.nn.Module
        Trained score model.
    init_ : torch.Tensor of shape (nsamples, ndim + 2)
        Initial states including noise-level and time columns.
    time_ : torch.Tensor of shape (nsamples,)
        Time values assigned to generated samples.
    L : int
        Number of noise levels.
    noise_lvl : float, default=0
        Noise level at which to stop the annealing schedule.
        Effective stopping level is ``max(min(sigma), noise_lvl)`` where
        ``sigma`` is the geometric noise sequence.

    Returns
    -------
    sol : torch tensor of shape (nsamples, ndim + 2)
        Generated states including auxiliary columns.
    """
    eps = 1e-4
    sol = init_.clone()
    nsamples = sol.shape[0]
    ndim = sol.shape[1] - 2
    sol[:, ndim + 1] = time_

    sigma = torch.as_tensor(geometric_sequence(L), dtype=sol.dtype, device=sol.device)
    min_sigma = torch.min(sigma)
    target_noise = torch.maximum(min_sigma, torch.as_tensor(noise_lvl, dtype=sol.dtype, device=sol.device))
    stop_k = int(torch.argmin(torch.abs(sigma - target_noise)).item())

    for k in range(0, stop_k + 1):
        alpha = eps * ((sigma[k] ** 2) / (sigma[L - 1] ** 2))
        sol[:, ndim] = sigma[k]

        for _ in range(0, maxiter):
            z = torch.randn((nsamples, ndim), dtype=sol.dtype, device=sol.device)
            guru = infNet(sol)
            sol[:, 0:ndim] = sol[:, 0:ndim] + 0.5 * alpha * guru + torch.sqrt(alpha) * z

    sol = torch.reshape(sol, (nsamples, ndim + 2))

    return sol


def freeze_dsm_score(
    net,
    noise_lvl=0.1,
):
    """Freeze the DSM noise-conditioning variable for flow inference.

    Parameters
    ----------
    net : torch.nn.Module
        Noise-conditional score network taking ``(x, sigma, t)`` inputs.
    noise_lvl : float, default=0
        Noise level used to freeze the sigma-conditioning input.

    Returns
    -------
    frozen : torch.nn.Module
        Wrapper taking ``(x, t)`` and injecting the fixed sigma value before
        the time coordinate.
    """
    return FreezeVarDNN(net=net, var_index=-1, var_value=noise_lvl)
