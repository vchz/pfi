"""Data reshaping and snapshot conversion helpers for PFI estimators."""

import numpy as np
import torch


def subsample_shuffle(
    snaps,
):
    """Shuffle each snapshot and subsample to a common sample count.

    Parameters
    ----------
    snaps : list of array-like, length n_snaps
        ``snaps[k]`` has shape ``(n_k, ndim)``.

    Returns
    -------
    dist : list of ndarray, length n_snaps
        Shuffled snapshots with common shape ``(n_min, ndim)`` where
        ``n_min = min_k n_k``.
    """
    nsamples = min(s.shape[0] for s in snaps)
    ndim = snaps[0].shape[1]
    dist = []

    for s in snaps:
        ind = np.arange(s.shape[0])
        np.random.shuffle(ind)
        dist.append(s[ind[:nsamples], :ndim])

    return dist


def X_from_snapshots(
    snaps,
    times,
):
    """Stack snapshots into a single array with appended time column.

    Parameters
    ----------
    snaps : list of ndarray, length n_snaps
        ``snaps[k]`` has shape ``(n_k, ndim)``.
    times : array-like of shape (n_snaps,)
        Snapshot times aligned with ``snaps``.

    Returns
    -------
    X : ndarray of shape (sum_k n_k, ndim + 1)
        Concatenated dataset where last column stores time.
    """
    X_list = []
    for k, t in enumerate(times):
        xk = snaps[k]
        tk = t * np.ones((xk.shape[0], 1))
        X_list.append(np.hstack([xk, tk]))

    return np.vstack(X_list)


def snapshots_from_X(
    X,
):
    """Split a time-augmented dataset into per-time snapshots.

    Parameters
    ----------
    X : array-like of shape (n_samples_total, ndim + 1)
        Input matrix with time stored in the last column.

    Returns
    -------
    snaps : list of torch.Tensor, length n_unique_times
        ``snaps[k]`` has shape ``(n_k, ndim)``.
    times : torch.Tensor of shape (n_unique_times,)
        Sorted unique times found in ``X``.
    """
    x = torch.tensor(X, dtype=torch.float32)
    t = x[:, -1]
    times = torch.unique(t)
    times, _ = torch.sort(times)

    snaps = []
    for ti in times:
        snaps.append(x[t == ti][:, :-1])

    return snaps, times
