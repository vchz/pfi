"""Optimal-transport coupling and trajectory sampling utilities."""

import numpy as np
import torch
import ot

from ...utils.data import subsample_shuffle


def compute_pairwise_ot_plans(
    samples,
    method="exact",
    epsilon=1e-2,
):
    """Compute pairwise transport plans between consecutive snapshots.

    Parameters
    ----------
    samples : torch.Tensor of shape (k_plus_1, b, ndim)
        Snapshot samples with equal sample count ``b``.
    method : {'exact', 'sinkhorn'}, default='exact'
        OT solver used for each consecutive pair.
    epsilon : float, default=1e-2
        Entropic regularization when ``method='sinkhorn'``.

    Returns
    -------
    pi_list : list of length k_plus_1 - 1
        Each entry is an ndarray of shape ``(b, b)`` containing an OT plan.
    """
    k_plus_1, b, _ = samples.shape
    pi_list = []
    uniform_weights = torch.ones(b) / b

    for k in range(k_plus_1 - 1):
        x = samples[k]
        y = samples[k + 1]
        c = ot.dist(x, y, metric="sqeuclidean")

        if method == "sinkhorn":
            pi = ot.sinkhorn(uniform_weights, uniform_weights, c, reg=epsilon)
        elif method == "exact":
            pi = ot.emd(uniform_weights, uniform_weights, c)
        else:
            raise ValueError("Unknown OT method")

        pi_list.append(pi)

    return pi_list


def sample_trajectory(
    pi_list,
    num_samples=1,
):
    """Sample discrete index trajectories from chained OT plans.

    Parameters
    ----------
    pi_list : list of length k
        Pairwise OT plans. Each plan has shape ``(b, b)``.
    num_samples : int, default=1
        Number of trajectories to sample.

    Returns
    -------
    trajectories : ndarray of shape (num_samples, k + 1)
        Sampled index trajectories through snapshots.
    """
    k = len(pi_list) + 1
    b = pi_list[0].shape[0]
    trajectories = np.zeros((num_samples, k), dtype=int)

    for n in range(num_samples):
        i_k = np.random.choice(b)
        trajectories[n, 0] = i_k
        for t in range(k - 1):
            probs = pi_list[t][i_k]
            probs = probs / probs.sum()
            i_kp1 = np.random.choice(b, p=probs)
            trajectories[n, t + 1] = i_kp1
            i_k = i_kp1

    return trajectories


def get_sample_paths(
    samples,
    trajectories,
    num_paths,
):
    """Gather full sample paths from trajectory indices.

    Parameters
    ----------
    samples : torch.Tensor of shape (k_plus_1, b, ndim)
        Snapshot samples.
    trajectories : ndarray of shape (num_paths, k_plus_1)
        Index trajectories.
    num_paths : int
        Number of paths to extract.

    Returns
    -------
    sample_paths : torch.Tensor of shape (k_plus_1, num_paths, ndim)
        Reconstructed paths.
    """
    _, k_plus_1 = trajectories.shape
    d = samples.shape[-1]
    sample_paths = torch.zeros((k_plus_1, num_paths, d), dtype=torch.float32)

    for t in range(k_plus_1):
        sample_paths[t, :, :] = samples[t][trajectories[:, t]]

    return sample_paths


def MMOT_trajectories(
    dist,
    nb=1,
    device="cpu",
):
    """Construct multi-marginal OT trajectories from snapshot lists.

    Parameters
    ----------
    dist : list of array-like, length nsnaps
        ``dist[k]`` has shape ``(n_k, ndim)``.
    nb : int, default=1
        Number of mini-batches used for OT stitching.
    device : str or torch.device, default='cpu'
        Target device for returned trajectories.

    Returns
    -------
    dist : torch.Tensor of shape (nsnaps, nsamples, ndim)
        Subsampled and stacked snapshots.
    batch_ot_samples : torch.Tensor of shape (nsnaps, nsamples, ndim)
        OT-coupled trajectories across snapshots.
    """
    dist = torch.stack(subsample_shuffle(dist), dim=0)
    nsamples = dist.shape[1]
    batch_ot_samples = torch.zeros_like(dist)

    # Note that we might loose some data points here (the remainder)
    bs = int(nsamples / nb)
    ind = np.arange(nsamples)
    for k in range(nb):
        np.random.shuffle(ind)
        chunk = ind[:bs]
        chunk_dist = dist[:, chunk, :]
        pi_list = compute_pairwise_ot_plans(chunk_dist, method="exact")
        trajectories = sample_trajectory(pi_list, num_samples=len(chunk))
        chunk_paths = get_sample_paths(chunk_dist, trajectories, len(chunk))
        batch_ot_samples[:, k * bs:(k + 1) * bs, :] = chunk_paths

    return dist, batch_ot_samples.to(device)
