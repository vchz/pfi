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


def load_data(
    path,
    nsamples,
    genes,
    time_key,
    cell_type_key,
    seed=0,
):
    """Load AnnData snapshots and sample a fixed number of cells per time.

    Parameters
    ----------
    path : str
        Path to the ``.h5ad`` dataset.
    nsamples : int
        Number of cells sampled at each time point.
    genes : list of str
        Selected genes used to build expression snapshots.
    time_key : str
        Name of the observation column storing time labels.
    cell_type_key : str
        Name of the observation column storing cell-type labels.
    seed : int, default=0
        Random seed for snapshot subsampling.

    Returns
    -------
    samples : ndarray of shape (n_times, nsamples, n_genes)
        Subsampled expression snapshots.
    unique_times : ndarray of shape (n_times,)
        Unique times present in the dataset.
    ind_array : ndarray of shape (n_times, nsamples)
        Encoded cell-type labels for sampled cells.
    cell_types : pandas.Series
        Full cell-type annotation column.
    """
    n_genes = len(genes)
    import scanpy as sc

    adata = sc.read_h5ad(path)

    unique_times = np.asarray(adata.obs[time_key].unique())
    samples = np.zeros((len(unique_times), nsamples, n_genes), dtype=np.float32)

    cell_type_categories = list(adata.obs[cell_type_key].cat.categories)
    cell_type_to_int = {ct: i for i, ct in enumerate(cell_type_categories)}
    ind_array = np.zeros((len(unique_times), nsamples), dtype=int)
    rng = np.random.default_rng(seed)

    for k, time_point in enumerate(unique_times):
        cells_at_time = adata[adata.obs[time_key] == time_point]
        expr = cells_at_time[:, genes].X
        expr = expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr)

        n_cells = expr.shape[0]
        if n_cells >= nsamples:
            selected = rng.choice(n_cells, size=nsamples, replace=False)
        else:
            selected = rng.choice(n_cells, size=nsamples, replace=True)

        cell_types = cells_at_time.obs[cell_type_key].iloc[selected].values
        ind_array[k, :] = [cell_type_to_int[ct] for ct in cell_types]
        samples[k, :, :] = expr[selected, :]

    return samples, unique_times, ind_array, adata.obs[cell_type_key]
