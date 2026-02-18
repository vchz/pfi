"""DeepRUOTv2 flow estimator wrapper."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .._base import FlowModel


def _deep_update(dst, src):
    """Recursively merge ``src`` into ``dst`` in-place.

    Parameters
    ----------
    dst : dict
        Destination dictionary updated in-place.
    src : dict
        Source dictionary.

    Returns
    -------
    dst : dict
        Updated destination dictionary.
    """
    for k, v in src.items():
        if isinstance(v, dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


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
    Ndim = X.shape[1] - 1
    gene_cols = [f"x{i}" for i in range(1, Ndim + 1)]
    raw_t = X[:, -1]
    shifted_t = raw_t - np.min(raw_t)
    df = pd.DataFrame(X[:, :Ndim], columns=gene_cols)
    df.insert(0, "samples", shifted_t)
    df.to_csv(csv_path, index=False)
    return df


class DeepRUOTv2Flow(torch.nn.Module):
    """DeepRUOTv2 force wrapper.

    Parameters
    ----------
    v_net : torch.nn.Module
        Drift sub-network from DeepRUOTv2.
    score_model : torch.nn.Module
        Score network providing ``compute_gradient``.
    sigma : float
        Diffusion level used by legacy SDE.
    Ndim : int
        State dimension.
    """

    def __init__(self, v_net, score_model, sigma, Ndim):
        """Initialize the DeepRUOTv2 force wrapper."""
        super().__init__()
        self.v_net = v_net
        self.score_model = score_model
        self.sigma = sigma
        self.Ndim = Ndim

    def forward(self, Xtrain, stoch=False):
        """Compute DeepRUOTv2 force.

        Parameters
        ----------
        Xtrain : torch.Tensor of shape (batch_size, Ndim + 1)
            Input states with time in the last column.
        stoch : bool, default=False
            If ``True``, returns ``(force, sigma * eps)``.

        Returns
        -------
        force : torch.Tensor of shape (batch_size, Ndim)
            Deterministic force.
        tuple : (force, noise)
            Stochastic pair when ``stoch=True``.
        """
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


class DeepRUOTv2(FlowModel):
    """DeepRUOTv2 estimator.

    Parameters
    ----------
    solver : str, default='fm'
        Solver backend name (unused by this wrapper, kept for API parity).
    solver_kwargs : dict or None, default=None
        Wrapper options. Supported keys include
        ``exp_name`` (str) and ``config`` (dict).
    device : str or torch.device, default='cpu'
        Device used for fit and inference.
    """

    def fit(self, X, y=None):
        """Fit DeepRUOTv2 from time-augmented data.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Input matrix with state variables in the first ``ndim`` columns
            and biological time in the last column.
        y : None, default=None
            Unused. Included for estimator API compatibility.

        Returns
        -------
        self : DeepRUOTv2
            Fitted estimator.

        Notes
        -----
        This method writes a CSV file in ``.deepruotv2/train.csv`` under the
        current working directory, then runs DeepRUOTv2 training/evaluation
        through ``TrainingPipeline``.

        Fitted attributes created by this method include:
        ``Ndim_``, ``workdir_``, ``csv_path_``, ``train_df_``, ``config_``,
        ``pipeline_``, ``pretrain_losses_``, ``final_losses_``,
        ``evaluation_``, ``f_net_``, ``score_net_``, ``sigma_``,
        ``flow_``, ``growth_``, and ``times_``.
        """
        from train_RUOT import TrainingPipeline
        from DeepRUOT.utils import load_and_merge_config

        X = np.asarray(X)
        self.Ndim_ = X.shape[1] - 1

        self.workdir_ = Path.cwd() / ".deepruotv2"
        self.workdir_.mkdir(parents=True, exist_ok=True)
        self.csv_path_ = self.workdir_ / "train.csv"
        self.train_df_ = to_deepruot_csv(X, self.csv_path_)

        config = load_and_merge_config(None)
        config["device"] = str(self.device)
        config["exp"]["output_dir"] = str(self.workdir_)
        config["exp"]["name"] = self.solver_kwargs.get("exp_name", "deepruotv2")
        config["data"]["file_path"] = str(self.csv_path_)
        config["data"]["dim"] = self.Ndim_
        config["model"]["in_out_dim"] = self.Ndim_

        user_cfg = self.solver_kwargs.get("config", {})
        _deep_update(config, user_cfg)
        self.config_ = config
        
        self.pipeline_ = TrainingPipeline(config)
        self.pretrain_losses_, self.final_losses_ = self.pipeline_.train()
        self.evaluation_ = self.pipeline_.evaluate()

        self.f_net_ = self.pipeline_.f_net
        self.score_net_ = self.pipeline_.sf2m_score_model
        self.sigma_ = float(config["score_train"]["sigma"])

        self.flow_ = DeepRUOTv2Flow(
            self.f_net_.v_net,
            self.score_net_,
            self.sigma_,
            self.Ndim_,
        ).to(self.device)
        self.growth_ = self.f_net_.g_net
        self.times_ = np.sort(self.train_df_["samples"].unique())
        self.flow_ = self.flow_.eval()
        return self
