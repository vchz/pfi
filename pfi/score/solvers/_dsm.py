"""Denoising score-matching (DSM) solvers and sampling utilities."""

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from ...utils.nns import FastTensorDataLoader, loss_grad_std
from ...utils.data import subsample_shuffle


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
    device,
):
    """Generate noisy DSM training inputs and normalization statistics.

    Parameters
    ----------
    Dist : torch.Tensor or ndarray of shape (nsnaps, nsamples, ndim)
        Snapshot samples.
    ndim : int
        State dimension.
    tp : array-like of shape (nsnaps,)
        Snapshot times.
    L : int
        Number of noise levels.
    nsamples : int
        Number of samples per snapshot.
    nsnaps : int
        Number of snapshots.
    device : str or torch.device
        Target device for returned tensors.

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
    sigma : ndarray of shape (L,)
        Noise schedule.
    """
    sigma = geometric_sequence(L)
    transform_data = np.zeros((nsnaps, L, nsamples, ndim + 2))

    for tind in range(nsnaps):
        for i in range(0, L):
            for j in range(0, nsamples):
                mean = Dist[tind, j, 0:ndim]
                cov = (sigma[i] ** 2) * np.eye(ndim)
                transform_data[tind, i, j, 0:ndim] = np.random.multivariate_normal(mean, cov)
                transform_data[tind, i, j, ndim] = sigma[i]
                transform_data[tind, i, j, ndim + 1] = tp[tind]

    x_train = torch.tensor(transform_data, dtype=torch.float32, requires_grad=True, device=device)
    x_data = torch.tensor(Dist, dtype=torch.float32, device=device)

    x_mean = torch.zeros((ndim + 2,))
    x_std = torch.ones((ndim + 2,))
    for i in range(0, ndim + 2):
        x_mean[i] = torch.tensor(
            np.mean(transform_data[:, :, :, i].flatten(), axis=0, keepdims=True),
            dtype=torch.float32,
        )
        x_std[i] = torch.tensor(
            np.std(transform_data[:, :, :, i].flatten(), axis=0, keepdims=True),
            dtype=torch.float32,
        )

    x_mean = x_mean[np.newaxis, :]
    x_std = x_std[np.newaxis, :]

    return x_train, x_data, x_mean, x_std, sigma


def DSM_(
    dist,
    times,
    net,
    L=10,
    n_epochs=2000,
    bs=None,
    adp_flag=0,
    lr=1e-4,
    device="cpu",
    verbose=True,
):
    """Train a score network with denoising score matching.

    Parameters
    ----------
    dist : list of torch.Tensor, length nsnaps
        ``dist[k]`` has shape ``(n_k, ndim)``.
    times : torch.Tensor of shape (nsnaps,)
        Snapshot times.
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
    device : str or torch.device, default='cpu'
        Training device.
    verbose : bool, default=True
        If ``True``, show progress bars and diagnostics.

    Returns
    -------
    net : torch.nn.Module
        Trained score network.
    loss_hist : list of float
        Epoch-wise training loss values.
    """
    dist = torch.stack(subsample_shuffle(dist), dim=0)

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
        device,
    )

    net.set_scales(x_mean.to(device), x_std.to(device))

    xtrain = torch.tensor(x_train, dtype=torch.float32).to(device)
    xdata = torch.tensor(x_data, dtype=torch.float32).to(device)

    loader = FastTensorDataLoader(
        torch.permute(xtrain, (2, 0, 1, 3)),
        torch.permute(xdata, (1, 0, 2)),
        batch_size=bs,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(list(net.parameters()), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[2500, 6500, 8500], gamma=0.1)

    c_ = torch.ones((nsnaps,), dtype=torch.float32, device=device)
    alpha_ann = 0.5
    adapt_int = 10
    weight_decay = 1e-4

    sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=x_train.device)
    sigma_reshaped = sigma_tensor.view(L, 1, 1)
    pbar = tqdm(range(n_epochs), desc="DSM", dynamic_ncols=True, disable=not verbose)
    loss_hist = []

    for epoch in pbar:
        for _, (xbatch, ybatch) in enumerate(loader):
            optimizer.zero_grad()

            xbatch = xbatch.clone().detach().requires_grad_(True)
            ybatch = ybatch.clone().detach().requires_grad_(True)

            xbatch = torch.permute(xbatch, (1, 2, 0, 3))
            ybatch = torch.permute(ybatch, (1, 0, 2))

            uhat = net(xbatch)
            lcomp = torch.zeros((nsnaps,), device=device)
            std_ = torch.zeros((nsnaps,), device=device)

            for tind in range(nsnaps):
                u_pred = uhat[tind]
                x_t = xbatch[tind, :, :, :ndim]
                x_data = ybatch[tind, :, :ndim][np.newaxis, :, :]

                u_true = (x_t - x_data) / (sigma_reshaped ** 2)
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
            if epoch % 500 == 0:
                tqdm.write(f"epoch: {epoch} c_: {c_.detach().cpu().numpy()}")

    return net, loss_hist


def generate_data_DSM(
    maxiter,
    infNet,
    nsamples,
    init_,
    time_,
    L,
    ndim,
    device,
):
    """Sample from a trained DSM model using annealed Langevin dynamics.

    Parameters
    ----------
    maxiter : int
        Number of Langevin steps per noise level.
    infNet : torch.nn.Module
        Trained score model.
    nsamples : int
        Number of samples to generate.
    init_ : ndarray of shape (nsamples, ndim + 2)
        Initial states including noise-level and time columns.
    time_ : float
        Time value assigned to all generated samples.
    L : int
        Number of noise levels.
    ndim : int
        State dimension.
    device : str or torch.device
        Device used for generation.

    Returns
    -------
    sol : ndarray of shape (nsamples, ndim + 2)
        Generated states including auxiliary columns.
    gen_mean : torch.Tensor of shape (ndim,)
        Mean of generated state coordinates.
    """
    eps = 1e-4
    sol = torch.tensor(init_, dtype=torch.float32).to(device)
    sol[:, ndim + 1] = time_

    sigma = geometric_sequence(L)

    for k in range(0, L):
        alpha = eps * ((sigma[k] ** 2) / (sigma[L - 1] ** 2))
        sol[:, ndim] = sigma[k]

        for _ in range(0, maxiter):
            z = torch.normal(0, 1, size=(nsamples, ndim)).to(device)
            guru = infNet(sol)
            sol[:, 0:ndim] = sol[:, 0:ndim] + 0.5 * alpha * guru + np.sqrt(alpha) * z

    gen_mean = torch.mean(sol, axis=0)[0:ndim]
    sol = sol.cpu().data.numpy().reshape(nsamples, ndim + 2)

    return sol, gen_mean
