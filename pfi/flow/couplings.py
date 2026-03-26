"""Pairwise OT coupling and trajectory stitching utilities."""

import torch
import ot


def _compute_pairwise_OT_plan(
    x,
    y,
    *,
    mass_x,
    mass_y,
    reg=0.1,
    reg_m=None,
    method="sinkhorn",
):
    """Compute a pairwise OT/UOT plan between two snapshots."""
    nx = x.shape[0]
    ny = y.shape[0]
    c = ot.dist(x, y, metric="sqeuclidean")

    if reg_m is None:
        a = torch.full((nx,), 1.0 / nx, dtype=x.dtype, device=x.device)
        b = torch.full((ny,), 1.0 / ny, dtype=y.dtype, device=y.device)
        if method == "exact":
            plan = ot.emd(a, b, c, numItermax=1000000)
        else:
            plan = ot.sinkhorn(
                a,
                b,
                c,
                reg=reg,
                method=method,
                reg_type="kl",
                stopThr=1e-6,
                numItermax=1000000,
            )
    else:
        a = torch.full((nx,), mass_x / nx, dtype=x.dtype, device=x.device)
        b = torch.full((ny,), mass_y / ny, dtype=y.dtype, device=y.device)
        plan = ot.sinkhorn_unbalanced(
            a,
            b,
            c,
            reg=reg,
            reg_m=reg_m,
            method=method,
            reg_type="kl",
            stopThr=1e-7,
            numItermax=500000,
        )

    if torch.isclose(plan.sum(), torch.as_tensor(0.0, dtype=plan.dtype, device=plan.device), atol=1e-8):
        raise ValueError("OT plan has near-zero mass; adjust solver/regularization settings.")
    return plan / plan.sum()


def _sample_trajectory(plan_list, num_samples):
    """Sample index trajectories from chained pairwise plans."""
    device = plan_list[0].device
    n0 = plan_list[0].shape[0]
    traj = [torch.randint(0, n0, (num_samples,), device=device)]

    for k in range(len(plan_list)):
        probs = plan_list[k][traj[k]]
        probs = probs / probs.sum(dim=1, keepdim=True)
        next_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        traj.append(next_idx)
    return traj


def pairwise_OT(
    dist,
    nb=1,
    reg=0.1,
    reg_m=None,
    method="sinkhorn",
):
    """Build stitched trajectories from consecutive pairwise OT plans.

    Parameters
    ----------
    dist : list of torch.Tensor, length nsnaps
        ``dist[k]`` has shape ``(n_k, ndim)`` on a common device.
    nb : int, default=1
        Number of mini-batches used for stitching.
    reg : float, default=0.1
        Entropic regularization strength.
    reg_m : float or None, default=None
        If ``None``, uses balanced pairwise OT with equal-size subsampling and
        returns ``psi_traj = 1``. If not ``None``, uses unbalanced OT.
    method : str, default='sinkhorn'
        Backend passed to POT OT/UOT solvers.

    Returns
    -------
    batch_ot_samples : torch.Tensor of shape (nsnaps, ntraj_total, ndim)
        Sampled and stitched trajectories.
    psi_traj : torch.Tensor of shape (nsnaps, ntraj_total)
        Per-snapshot trajectory weights.
    """
    nsnaps = len(dist)
    device = dist[0].device
    dtype = dist[0].dtype

    batch_paths = []
    batch_psi = []

    if reg_m is None:
        nsamples = min(s.shape[0] for s in dist)
        bs = max(1, nsamples // nb)
        bs_list = [bs] * nsnaps
        masses = None
    else:
        bs_list = [s.shape[0] // nb for s in dist]
        masses = [s.shape[0] for s in dist]
    ntraj = min(bs_list)

    for _ in range(nb):
        chunks = []
        for k in range(nsnaps):
            idx = torch.randperm(dist[k].shape[0], device=dist[k].device)[: bs_list[k]]
            chunks.append(dist[k].index_select(0, idx))

        plan_list = []
        for k in range(nsnaps - 1):
            plan_list.append(
                _compute_pairwise_OT_plan(
                    chunks[k],
                    chunks[k + 1],
                    mass_x=None if masses is None else masses[k],
                    mass_y=None if masses is None else masses[k + 1],
                    reg=reg,
                    reg_m=reg_m,
                    method=method,
                )
            )

        traj = _sample_trajectory(plan_list, num_samples=ntraj)
        paths_k = torch.zeros((nsnaps, ntraj, dist[0].shape[1]), dtype=dtype, device=device)
        for k in range(nsnaps):
            paths_k[k] = chunks[k].index_select(0, traj[k])

        if reg_m is None:
            psi_k = torch.ones((nsnaps, ntraj), dtype=dtype, device=device)
        else:
            psi_k = torch.zeros((nsnaps, ntraj), dtype=dtype, device=device)
            probs = torch.eye(ntraj, dtype=dtype, device=device)
            for k in range(nsnaps):
                if k > 0:
                    gamma = plan_list[k - 1]
                    probs = probs @ (gamma / gamma.sum(dim=1, keepdim=True))
                denom = probs.mean(dim=0).index_select(0, traj[k])
                psi_k[k] = masses[k] / bs_list[k] / denom

        batch_paths.append(paths_k)
        batch_psi.append(psi_k)

    batch_ot_samples = torch.cat(batch_paths, dim=1)
    psi_traj = torch.cat(batch_psi, dim=1)
    return batch_ot_samples, psi_traj


__all__ = [
    "pairwise_OT",
]
