"""DeepRUOTv2 external solver."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ....utils.data import deep_dict_update


def to_deepruot_csv(X, csv_path):
    """Export time-augmented samples to DeepRUOTv2 CSV format.

    Parameters
    ----------
    X : array-like of shape (n_samples, ndim + 1)
        Input matrix with time in the last column.
    csv_path : str or pathlib.Path
        Output CSV path.

    Returns
    -------
    df : pandas.DataFrame
        Written dataframe with columns ``samples, x1, ..., xN`` where
        ``samples`` stores biological times shifted so the minimum time is 0.
    """
    X = np.asarray(X)
    ndim = X.shape[1] - 1
    gene_cols = [f"x{i}" for i in range(1, ndim + 1)]
    raw_t = X[:, -1]
    shifted_t = raw_t - np.min(raw_t)
    df = pd.DataFrame(X[:, :ndim], columns=gene_cols)
    df.insert(0, "samples", shifted_t)
    df.to_csv(csv_path, index=False)
    return df


class _DeepRUOTv2Flow(torch.nn.Module):
    """Private DeepRUOTv2 force wrapper."""

    def __init__(self, v_net, score_model, ndim, sigma=1.0):
        """Initialize the DeepRUOTv2 force wrapper."""
        super().__init__()
        self.v_net = v_net
        self.score_model = score_model
        self.sigma = sigma
        self.Ndim = ndim

    def forward(self, Xtrain, stoch=False):
        """Compute DeepRUOTv2 force."""
        t = Xtrain[:, -1:]
        x = Xtrain[:, : self.Ndim]
        drift = self.v_net(t, x)

        if self.sigma > 0:
            with torch.enable_grad():
                xg = x.clone().requires_grad_(True)
                tg = t.clone()
                drift = self.v_net(tg, xg)
                force = drift + self.score_model.compute_gradient(tg, xg)
        else:
            force = drift

        if stoch:
            return force, self.sigma * torch.randn_like(x)
        return force


def DeepRUOTv2_(dist, times, flow_model, growth_model=None, **solver_kwargs):
    """Fit DeepRUOTv2 from snapshot tensors.

    Parameters
    ----------
    dist : list of torch.Tensor
        Snapshot tensors, each of shape ``(n_k, ndim)``.
    times : torch.Tensor of shape (n_snaps,)
        Snapshot times.
    flow_model : None
        Unused placeholder for API compatibility.
    growth_model : None, optional
        Unused placeholder for API compatibility.
    **solver_kwargs : dict
        DeepRUOTv2 solver options. Supported keys include
        ``exp_name`` and ``config``.

    Returns
    -------
    flow : torch.nn.Module
        Fitted DeepRUOTv2 flow wrapper.
    growth : torch.nn.Module or None
        Fitted DeepRUOTv2 growth model.
    loss_hist : ndarray
        Training loss history.
    """
    from train_RUOT import TrainingPipeline
    from DeepRUOT.utils import load_and_merge_config

    _ = flow_model, growth_model

    ndim = dist[0].shape[1]
    device = dist[0].device

    workdir = Path.cwd() / ".deepruotv2"
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "train.csv"

    t0 = float(times.min().item())
    rows = []
    for k, snap in enumerate(dist):
        xk = snap.detach().cpu().numpy()
        tk = float(times[k].item()) - t0
        rows.append(np.hstack([tk * np.ones((xk.shape[0], 1)), xk]))
    X = np.vstack(rows)
    train_df = to_deepruot_csv(X, csv_path)

    config = load_and_merge_config(None)
    config["device"] = str(device)
    config["exp"]["output_dir"] = str(workdir)
    config["exp"]["name"] = solver_kwargs.get("exp_name", "deepruotv2")
    config["data"]["file_path"] = str(csv_path)
    config["data"]["dim"] = ndim
    config["model"]["in_out_dim"] = ndim

    user_cfg = solver_kwargs.get("config", {})
    config = deep_dict_update(config, user_cfg)

    pipeline = TrainingPipeline(config)
    _, final_losses = pipeline.train()
    _ = pipeline.evaluate()

    f_net = pipeline.f_net
    score_net = pipeline.sf2m_score_model
    sigma = float(config["score_train"]["sigma"])

    flow = _DeepRUOTv2Flow(
        f_net.v_net,
        score_net,
        ndim,
        sigma,
    ).to(device).eval()

    growth = f_net.g_net
    if growth is not None:
        growth = growth.to(device).eval()

    return flow, growth, np.asarray(final_losses)
