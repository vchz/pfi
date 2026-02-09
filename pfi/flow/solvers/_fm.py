"""Flow-matching solver and helper routines."""

import numpy as np
import torch
from tqdm import tqdm

from ...utils.nns import loss_grad_std
from ...utils.data import subsample_shuffle
from ..interpolants import MMOT_trajectories, select_best_lambda, ChebyshevInterpolant


def interpolate_old2new(
    Dist,
    old_nodes,
    new_nodes,
):
    """Interpolate trajectories from one temporal grid to another.

    Parameters
    ----------
    Dist : torch.Tensor of shape (n_old, batch_size, ndim)
        Input trajectories at ``old_nodes``.
    old_nodes : torch.Tensor of shape (batch_size, n_old)
        Original time nodes.
    new_nodes : torch.Tensor of shape (batch_size, n_new)
        Target time nodes.

    Returns
    -------
    x_interp : torch.Tensor of shape (n_new, batch_size, ndim)
        Interpolated trajectories.
    """
    _, b, d = Dist.shape
    x = torch.permute(Dist, (1, 0, 2))

    x_interp = torch.stack(
        [
            torch.stack(
                [
                    torch.tensor(
                        np.interp(
                            new_nodes[i].cpu().numpy(),
                            old_nodes[i].cpu().numpy(),
                            x[i, :, j].cpu().numpy(),
                        ),
                        dtype=torch.float32,
                    )
                    for j in range(d)
                ],
                dim=1,
            )
            for i in range(b)
        ],
        dim=0,
    )

    return torch.permute(x_interp, (1, 0, 2))


def FM_(
    dist,
    times,
    interp,
    net,
    growth_model=None,
    fac=1,
    nb=1,
    n_epochs=2000,
    lr=1e-3,
    alpha_ann=0.5,
    device="cpu",
    verbose=True,
):
    """Train drift and optional growth models by flow matching.

    Parameters
    ----------
    dist : list of torch.Tensor, length nsnaps
        ``dist[k]`` has shape ``(n_k, ndim)``.
    times : torch.Tensor of shape (nsnaps,)
        Snapshot times.
    interp : object
        Interpolant implementing ``fit(nodes, data)`` and
        ``predict(nodes) -> (x_interp, dx_interp)``.
    net : torch.nn.Module
        Drift model mapping ``(batch_size, ndim + 1)`` to ``(batch_size, ndim)``.
    growth_model : torch.nn.Module or None, default=None
        Optional growth model mapping ``(batch_size, ndim + 1)`` to
        ``(batch_size, 1)``.
    fac : int, default=1
        Temporal upsampling factor for the uniform grid.
    nb : int, default=1
        Number of mini-batches used for OT stitching.
    n_epochs : int, default=2000
        Number of optimization epochs.
    lr : float, default=1e-3
        Learning rate.
    alpha_ann : float, default=0.5
        Exponential averaging factor for adaptive mass-loss weight.
    device : str or torch.device, default='cpu'
        Training device.
    verbose : bool, default=True
        If ``True``, show progress bars and diagnostics.

    Returns
    -------
    drift_net : torch.nn.Module
        Trained drift model.
    growth_model : torch.nn.Module or None
        Trained growth model (or ``None`` when disabled).
    loss_hist : list of float
        Epoch-wise total objective values.
    """
    
    dist_tensor = torch.stack(subsample_shuffle(dist), dim=0).to(device)
    x_mean = dist_tensor.mean(dim=(0, 1)).unsqueeze(0)
    x_std = dist_tensor.std(dim=(0, 1)).unsqueeze(0)
    net.set_scales(x_mean, x_std)

    dist_ot, batch_ot_samples = MMOT_trajectories(dist, nb=nb, device=device)
    nsnaps, nsamples, ndim = dist_ot.shape

    mass_vec = np.ones((nsnaps,))
    for k in range(1, nsnaps):
        mass_vec[k] = dist[k].shape[0] / dist[0].shape[0]
    mass_vec = torch.tensor(mass_vec, dtype=torch.float32, device=device)

    uniform_kind = torch.tensor(
        np.linspace(times[0].item(), times[-1].item(), fac * times.shape[0]),
        dtype=torch.float32,
    )
    data_nodes = times[None, :].repeat(batch_ot_samples.shape[1], 1)
    uniform_nodes = uniform_kind.to(device).expand(batch_ot_samples.shape[1], -1)
    batch_ot_samples_uniform = interpolate_old2new(batch_ot_samples, data_nodes, uniform_nodes).to(device)

    batch_size = int(nsamples / 4)
    print(nsamples, batch_size)
    uniform_batch = uniform_kind.to(device).expand(batch_size, -1)

    data_full = uniform_kind.to(device).expand(batch_ot_samples_uniform.shape[1], -1)

    if isinstance(interp, ChebyshevInterpolant):
        batch_ot_samples_uniform_ = torch.permute(batch_ot_samples_uniform[:, :, 0:ndim], (1, 0, 2))
        best_lam, _, _ = select_best_lambda(
            batch_ot_samples_uniform_,
            data_full,
            data_full,
            device,
            verbose=verbose,
        )
        interp.reg_ = best_lam

    drift_net = net.to(device)

    if growth_model is None:
        growth_model = None
        zero_growth = True
    else:
        growth_model = growth_model.to(device)
        zero_growth = False

    params = [{"params": drift_net.parameters(), "lr": lr}]
    if not zero_growth:
        params.append({"params": growth_model.parameters(), "lr": lr})

    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 1500, 8000, 15000], gamma=0.1)

    dt = times[1] - times[0]
    lamb = 1.0
    loss_hist = []

    pbar = tqdm(range(n_epochs), desc="FM", dynamic_ncols=True, disable=not verbose)
    for epoch in pbar:
        optimizer.zero_grad()

        if zero_growth:
            growth_ = torch.zeros_like(batch_ot_samples_uniform[..., 0])
        else:
            growth_ = growth_model(batch_ot_samples_uniform).squeeze(-1)

        log_mass = torch.zeros_like(growth_)
        log_mass[0] = np.log(1.0 / batch_ot_samples_uniform.shape[1])
        for k in range(1, fac * nsnaps):
            log_mass[k] = log_mass[k - 1] + 0.5 * (dt / fac) * (growth_[k] + growth_[k - 1])

        mass_ = torch.exp(log_mass)
        pred_mass = torch.sum(mass_, axis=1)

        Xtrain, ytrain, weights_ = compute_conditional_distributions(
            interp,
            batch_ot_samples_uniform,
            batch_size,
            mass_,
            uniform_batch,
            uniform_batch,
            device=device,
            sigma=0.001,
        )

        inp_ = Xtrain.clone()
        inp_.requires_grad_()
        fold = drift_net(inp_)

        res = (fold - ytrain).pow(2).sum(dim=1).reshape(batch_size, fac * nsnaps)
        weights_ = weights_ / torch.sum(weights_, axis=0)[None, :]
        wcfm_obj = torch.mean(weights_ * res, axis=0)
        mass_balance = (pred_mass[::fac] - mass_vec) ** 2

        l1 = torch.sum(wcfm_obj)
        l2 = torch.sum(mass_balance)

        if not zero_growth:
            if epoch % 10 == 0:
                with torch.no_grad():
                    std_l1 = loss_grad_std(l1, drift_net, device)
                    if zero_growth:
                        std_l2 = torch.tensor(1.0, device=device)
                    else:
                        std_l2 = loss_grad_std(l2, growth_model, device)
                    lamb_hat = std_l1 / std_l2
                    lamb = (1 - alpha_ann) * lamb + alpha_ann * lamb_hat

        total_loss = l1 + lamb * l2
        loss_hist.append(total_loss.item())

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose:
            pbar.set_postfix(loss=f"{total_loss.item():.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if epoch % 500 == 0:
                tqdm.write(
                    f"[epoch {epoch}] "
                    f"inferred mass={torch.sum(mass_, axis=1)[::fac].detach().cpu().numpy()} "
                    f"actual mass={mass_vec}"
                )

    return drift_net, growth_model, loss_hist


def compute_conditional_distributions(
    interp,
    Dist,
    batch_size,
    weights,
    nodes_fit,
    nodes_eval,
    device="cpu",
    sigma=0.001,
):
    """Build conditional training pairs for weighted flow matching.

    Parameters
    ----------
    interp : object
        Interpolant implementing ``fit`` and ``predict``.
    Dist : torch.Tensor of shape (n_nodes, nsamples, ndim)
        OT trajectories on a temporal grid.
    batch_size : int
        Number of trajectories sampled per iteration.
    weights : torch.Tensor of shape (n_nodes, nsamples)
        Per-trajectory importance weights.
    nodes_fit : torch.Tensor of shape (batch_size, n_nodes)
        Nodes used to fit interpolants.
    nodes_eval : torch.Tensor of shape (batch_size, n_eval)
        Nodes used to evaluate interpolants.
    device : str or torch.device, default='cpu'
        Device used for output tensors.
    sigma : float, default=0.001
        Gaussian perturbation added to interpolated positions.

    Returns
    -------
    xtrain : torch.Tensor of shape (batch_size * n_eval, ndim + 1)
        Inputs for drift training (state + time).
    ytrain : torch.Tensor of shape (batch_size * n_eval, ndim)
        Target velocities.
    weights : torch.Tensor of shape (batch_size, n_nodes)
        Sampled weights aligned with ``xtrain``/``ytrain``.
    """
    nsamples = Dist.shape[1]
    n_eval = nodes_eval.shape[1]
    ndim = Dist.shape[2]

    xtrain = torch.zeros((batch_size * n_eval, ndim + 1), dtype=torch.float32, device=device)
    ytrain = torch.zeros((batch_size * n_eval, ndim), dtype=torch.float32, device=device)

    xind = torch.randint(0, nsamples, (batch_size,))
    Dist = torch.tensor(Dist, dtype=torch.float32, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    weights = torch.permute(weights[:, xind], (1, 0))
    Dist = torch.permute(Dist[:, xind, 0:ndim], (1, 0, 2))

    interp.fit(nodes_fit, Dist)
    x_interp, dx_interp = interp.predict(nodes_eval)

    xtrain[:, ndim] = nodes_eval.reshape(batch_size * n_eval)

    mut = x_interp.view(batch_size * n_eval, ndim)
    xtrain[:, 0:ndim] = mut + sigma * torch.randn_like(mut)
    ytrain[:, 0:ndim] = dx_interp.view(batch_size * n_eval, ndim)

    return xtrain, ytrain, weights
