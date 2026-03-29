"""Data helpers for PFI estimators and applications."""

from pathlib import Path
import gzip
import shutil
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import torch

DATA_URLS = {
    "natcomm": "https://zenodo.org/records/19237708/files/10XChromiumV3_10.1038_s41467-021-27159-x_10.5281_zenodo.5291737.h5ad.gz",
    "kaggle": "https://zenodo.org/records/19237708/files/CITE-seq_10.1126_science.adi8577_GSE305370.h5ad.gz",
}
PFI_DATA_FOLDER = "~/pfi_data"


def fetch(url):
    """Fetch a ``.h5ad`` or ``.h5ad.gz`` file into ``PFI_DATA_FOLDER``.

    Parameters
    ----------
    url : str
        Remote URL ending with ``.h5ad`` or ``.h5ad.gz``.

    Returns
    -------
    local_path : str
        Local file path under ``PFI_DATA_FOLDER``.
    """
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if not (filename.endswith(".h5ad") or filename.endswith(".h5ad.gz")):
        raise ValueError("fetch(url) expects a URL ending with .h5ad or .h5ad.gz.")

    data_root = Path(PFI_DATA_FOLDER).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    target = data_root / filename
    if filename.endswith(".h5ad.gz"):
        unzipped = data_root / filename[:-3]
        if unzipped.exists():
            print(f"[fetch] using cached file: {unzipped}")
            return str(unzipped)
    else:
        if target.exists():
            print(f"[fetch] using cached file: {target}")
            return str(target)

    if not target.exists():
        print(f"[fetch] downloading: {url}")
        print(f"[fetch] saving to: {target}")
        urlretrieve(url, target)

    if filename.endswith(".h5ad.gz"):
        unzipped = data_root / filename[:-3]
        if not unzipped.exists():
            print(f"[fetch] unzipping: {target} -> {unzipped}")
            with gzip.open(target, "rb") as fin, open(unzipped, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        print(f"[fetch] ready: {unzipped}")
        return str(unzipped)

    print(f"[fetch] ready: {target}")
    return str(target)


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
    normalize=False,
    plot_total_counts=False,
    min_tot_counts=0.0,
    max_tot_counts=100.0,
):
    """Load AnnData snapshots and sample a fixed number of cells per time.

    Cells are globally filtered by total counts across selected genes before
    per-time subsampling: keep cells with total counts in
    ``[min_tot_counts, max_tot_counts]``.

    Parameters
    ----------
    path : str
        Either a local path to the ``.h5ad``/``.h5ad.gz`` dataset, or one of
        ``{"natcomm", "kaggle"}`` to fetch from ``DATA_URLS``.
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
    normalize : bool, default=False
        If ``True``, normalize each cell by its total counts and rescale by
        ``1e4`` before gene selection and subsampling.
    plot_total_counts : bool, default=False
        If ``True``, plot per-time-point histograms of per-cell total counts
        (over selected genes) before filtering and subsampling.
    min_tot_counts : float, default=0.0
        Minimum total-count threshold (strict).
    max_tot_counts : float, default=100.0
        Maximum total-count threshold (strict).

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

    if path in DATA_URLS:
        path = fetch(DATA_URLS[path])
    adata = sc.read_h5ad(path)
    expr_full = adata.X
    expr_full = expr_full.toarray() if hasattr(expr_full, "toarray") else np.asarray(expr_full)

    if normalize:
        total_full = expr_full.sum(axis=1, keepdims=True)
        adata.X = (1e4 * expr_full) / (total_full + 1e-12)

    expr_all = adata[:, genes].X
    expr_all = expr_all.toarray() if hasattr(expr_all, "toarray") else np.asarray(expr_all)
    total_counts = expr_all.sum(axis=1)

    if plot_total_counts:
        import matplotlib.pyplot as plt

        times_all = np.asarray(adata.obs[time_key].unique())
        n_panels = len(times_all)
        ncols = min(4, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
        obs_times = np.asarray(adata.obs[time_key])

        for i, t in enumerate(times_all):
            ax = axes[i // ncols, i % ncols]
            counts_t = total_counts[obs_times == t]
            ax.hist(counts_t, bins=20)
            ax.axvline(min_tot_counts, color="red")
            ax.axvline(max_tot_counts, color="red")
            ax.set_title(f"{time_key}={t}")
            ax.set_xlabel("total counts")
            ax.set_ylabel("n cells")
            ax.set_yscale("log")

        for j in range(n_panels, nrows * ncols):
            axes[j // ncols, j % ncols].set_axis_off()

        plt.tight_layout()
        plt.show()


    keep_mask = (total_counts > min_tot_counts) & (total_counts < max_tot_counts)
    n_removed = int((~keep_mask).sum())
    print(
        f"[load_data] removed {n_removed} cells outside total-count range "
        f"[{min_tot_counts}, {max_tot_counts}] "
        f"({n_removed / adata.n_obs:.2%} of total cells) before subsampling."
    )
    adata = adata[keep_mask].copy()

    unique_times = np.asarray(adata.obs[time_key].unique())
    samples = np.zeros((len(unique_times), nsamples, n_genes), dtype=np.float32)

    cell_type_categories = list(adata.obs[cell_type_key].cat.categories)
    print(cell_type_categories)
    cell_type_to_int = {ct: i for i, ct in enumerate(cell_type_categories)}
    ind_array = np.zeros((len(unique_times), nsamples), dtype=int)
    rng = np.random.default_rng(seed)

    for k, time_point in enumerate(unique_times):
        cells_at_time = adata[adata.obs[time_key] == time_point]
        expr = cells_at_time[:, genes].X
        expr = expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr)
        ct_values = cells_at_time.obs[cell_type_key].values

        n_cells = expr.shape[0]

        if n_cells >= nsamples:
            selected = rng.choice(n_cells, size=nsamples, replace=False)
        else:
            print('there were less cells than requested at snapshot', k)
            selected = rng.choice(n_cells, size=nsamples, replace=True)

        cell_types = ct_values[selected]
        ind_array[k, :] = [cell_type_to_int[ct] for ct in cell_types]
        samples[k, :, :] = expr[selected, :]

    return samples, unique_times, ind_array, adata.obs[cell_type_key]


def deep_dict_update(base, updates):
    """Recursively update a dictionary with nested-dictionary support.

    Parameters
    ----------
    base : dict
        Base dictionary to update.
    updates : dict
        Dictionary containing override values.

    Returns
    -------
    out : dict
        Updated dictionary. Nested dictionaries are merged recursively.
    """
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_dict_update(out[key], value)
        else:
            out[key] = value
    return out



def evaluate_signed_network_auprc(Delta, M, symmetric=False):
    """Compute edge/positive/negative AUPRC for signed network recovery.

    Parameters
    ----------
    Delta : array-like of shape (n, n)
        Inferred response matrix.
    M : array-like of shape (n, n)
        Ground-truth signed matrix with entries in ``{-1, 0, 1}``.
    symmetric : bool, default=False
        If ``True``, evaluate only the strict upper triangle.
        If ``False``, evaluate all off-diagonal entries.

    Returns
    -------
    out : dict
        Dictionary with:
        - ``AP_edge``, ``AP_pos``, ``AP_neg``, ``AP_signed``
        - PR-curve arrays under ``PR_edge``, ``PR_pos``, ``PR_neg``
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    Delta = np.asarray(Delta, dtype=float)
    M = np.asarray(M, dtype=float)
    if Delta.shape != M.shape:
        raise ValueError("Delta and M must have the same shape.")
    if Delta.ndim != 2 or Delta.shape[0] != Delta.shape[1]:
        raise ValueError("Delta and M must be square matrices.")

    n = Delta.shape[0]
    mask = np.ones((n,n), dtype=bool)
    if symmetric:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        

    d = Delta[mask].reshape(-1)
    y = M[mask].reshape(-1)

    y_edge = (y != 0).astype(int)
    s_edge = np.abs(d)

    y_pos = (y == 1).astype(int)
    s_pos = d

    y_neg = (y == -1).astype(int)
    s_neg = -d

    def _pr(y_true, score):
        if np.sum(y_true) == 0:
            return np.nan, np.array([1.0]), np.array([0.0]), np.array([])
        ap = float(average_precision_score(y_true, score))
        prec, rec, thr = precision_recall_curve(y_true, score)
        return ap, prec, rec, thr

    ap_edge, p_edge, r_edge, t_edge = _pr(y_edge, s_edge)
    ap_pos, p_pos, r_pos, t_pos = _pr(y_pos, s_pos)
    ap_neg, p_neg, r_neg, t_neg = _pr(y_neg, s_neg)
    ap_signed = float(np.nanmean([ap_pos, ap_neg]))

    return {
        "AP_edge": ap_edge,
        "AP_pos": ap_pos,
        "AP_neg": ap_neg,
        "AP_signed": ap_signed,
        "PR_edge": {"precision": p_edge, "recall": r_edge, "thresholds": t_edge},
        "PR_pos": {"precision": p_pos, "recall": r_pos, "thresholds": t_pos},
        "PR_neg": {"precision": p_neg, "recall": r_neg, "thresholds": t_neg},
    }


def get_hsc_network(genes):
    """Build the HSC regulatory sign matrix aligned with a gene list.

    Parameters
    ----------
    genes : sequence of str
        Gene names defining matrix order.

    Returns
    -------
    M : ndarray of shape (n_genes, n_genes)
        Regulatory sign matrix with entries in ``{-1, 0, 1}`` where
        ``M[i, j]`` encodes the sign of the effect of gene ``i`` on gene ``j``.
    network_genes : list of str, optional
        Returned only when ``return_network_genes=True``.
    """
    genes = list(genes)
    network_genes = [
        "gata1",
        "gata2",
        "fli1",
        "spi1",
        "zfpm1",
        "klf1",
        "tal1",
        "cebpa",
        "jun",
        "erg",
        "gfi1",
    ]
    idx = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(genes), len(genes)), dtype=float)

    def set_edge(src, tgt, sign):
        if src in idx and tgt in idx:
            M[idx[src], idx[tgt]] = float(sign)

    # Edges from the HSC boolean-network rules (activation=+1, inhibition=-1).
    set_edge("gata1", "gata1", +1)
    set_edge("gata2", "gata1", +1)
    set_edge("fli1", "gata1", +1)
    set_edge("spi1", "gata1", -1)

    set_edge("gata2", "gata2", +1)
    set_edge("gata1", "gata2", -1)
    set_edge("zfpm1", "gata2", -1)
    set_edge("spi1", "gata2", -1)

    set_edge("gata1", "zfpm1", +1)

    set_edge("gata1", "klf1", +1)
    set_edge("fli1", "klf1", -1)

    set_edge("gata1", "fli1", +1)
    set_edge("klf1", "fli1", -1)

    set_edge("gata1", "tal1", +1)
    set_edge("spi1", "tal1", -1)

    set_edge("cebpa", "cebpa", +1)
    set_edge("gata1", "cebpa", -1)
    set_edge("zfpm1", "cebpa", -1)
    set_edge("tal1", "cebpa", -1)

    set_edge("cebpa", "spi1", +1)
    set_edge("spi1", "spi1", +1)
    set_edge("gata1", "spi1", -1)
    set_edge("gata2", "spi1", -1)

    set_edge("spi1", "jun", +1)
    set_edge("gfi1", "jun", -1)

    set_edge("spi1", "erg", +1)
    set_edge("jun", "erg", +1)
    set_edge("gfi1", "erg", -1)

    set_edge("cebpa", "gfi1", +1)
    set_edge("erg", "gfi1", -1)

    return M, network_genes


def assign_OT(x, y, method="exact", reg=0.1, sym=False, return_indices=False):
    """Compute an OT pairing from ``x`` to ``y``.

    Parameters
    ----------
    x : array-like of shape (n_x, d)
        Source samples.
    y : array-like of shape (n_y, d)
        Target samples.
    method : str, default="exact"
        OT solver method. If ``"exact"``, use ``ot.emd``. Otherwise use
        ``ot.sinkhorn`` with the provided method name.
    reg : float, default=0.1
        Entropic regularization used by ``ot.sinkhorn``.
    sym : bool, default=False
        If ``False``, pair each source sample with ``argmax`` over transport
        rows. If ``True``, use all nonzero transport entries via
        ``np.nonzero(pi)``.
    return_indices : bool, default=False
        If ``True``, also return the source and target indices of the
        assignment pairs.

    Returns
    -------
    xk : ndarray of shape (n_x, d)
        Source samples in pairing order.
    yk : ndarray of shape (n_x, d)
        Target samples paired to ``xk``.
    """
    import ot

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(y.shape[0]) / y.shape[0]
    M = ot.dist(x, y, metric="euclidean") ** 2
    if method == "exact":
        pi = ot.emd(a, b, M, numItermax=1000000)
    else:
        pi = ot.sinkhorn(a, b, M, reg=reg, method=method)

    if sym:
        i_idx, j_idx = np.nonzero(pi)
    else:
        i_idx = np.arange(x.shape[0])
        j_idx = np.argmax(pi, axis=1)
    if return_indices:
        return x[i_idx], y[j_idx], i_idx, j_idx
    return x[i_idx], y[j_idx]


def compute_correlations(x, y):
    """Compute paired summary metrics between two arrays.

    Parameters
    ----------
    x : array-like
        First paired array.
    y : array-like
        Second paired array.

    Returns
    -------
    metrics : dict
        Dictionary with ``pearson``, ``spearman``, ``energy_distance`` and
        ``rmse_l2`` entries. The energy distance is computed with
        ``geomloss.SamplesLoss('energy')`` on the full sample arrays.
    """
    import geomloss
    from scipy.stats import pearsonr, spearmanr

    x_full = np.asarray(x, dtype=np.float64)
    y_full = np.asarray(y, dtype=np.float64)
    x = x_full.reshape(-1)
    y = y_full.reshape(-1)

    pearson = pearsonr(x, y)[0]
    spearman = spearmanr(x, y)[0]
    ed = geomloss.SamplesLoss("energy")(
        torch.as_tensor(x_full, dtype=torch.float32),
        torch.as_tensor(y_full, dtype=torch.float32),
    ).item()
    rmse = np.mean((x - y) ** 2)/np.mean(x**2)

    return {
        "pearson": pearson,
        "spearman": spearman,
        "energy_distance": ed,
        "rmse_l2": rmse,
    }
