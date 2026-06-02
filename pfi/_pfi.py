"""Composite PFI estimator and estimator factory helpers."""

import numpy as np
import torch
import torch.nn as nn

from .flow import FlowModel
from .flow.models import CLEFlow
from .score import ScoreModel
from .score.models import DSMScore
from .utils.nns import SpectralNormDNN


class PFI:
    """Composite estimator that fits score first, then flow."""

    def __init__(
        self,
        score_estimator,
        flow_estimator,
        fit_on_score_samples=False,
    ):
        self.score_estimator = score_estimator
        self.flow_estimator = flow_estimator
        self.fit_on_score_samples = fit_on_score_samples

    def fit(self, X, y=None):
        """Fit score estimator then flow estimator."""
        _ = y
        self.score_estimator.fit(X)
        self.flow_estimator.flow.score = self.score_estimator.model_
        X_flow = X
        if self.fit_on_score_samples:
            X_flow = self.score_estimator.sample(X)
        self.flow_estimator.fit(X_flow)
        return self

    def predict(self, X, stoch=False):
        """Predict flow values using the fitted flow estimator."""
        return self.flow_estimator.predict(X, stoch=stoch)

    def sample(self, X, Dt, dt=0.01, stoch=False):
        """Sample trajectories with the fitted flow estimator."""
        return self.flow_estimator.sample(X, Dt=Dt, dt=dt, stoch=stoch)

    def score(self, X, y, dt=0.01, stoch=False):
        """Compute flow-stage energy distance."""
        return self.flow_estimator.score(X, y, dt=dt, stoch=stoch)


def make_score_estimator(
    score_model=None,
    solver="dsm",
    solver_kwargs=None,
    device="cpu",
    seed=0,
):
    """Create a configured ``ScoreModel``.

    Parameters
    ----------
    score_model : torch.nn.Module or None, default=None
        Score model. If ``None``, a default ``DSMScore`` is created.
    solver : str, default='dsm'
        Score solver name.
    solver_kwargs : dict or None, default=None
        Extra kwargs passed to the score solver.
        The score solver reads optimization settings from ``solver_kwargs``
        (e.g. ``n_epochs``, ``lr``).
    device : str or torch.device, default='cpu'
        Device used by the score estimator.
    seed : int, default=0
        Random seed used when the default score model is built.
    """
    if score_model is None:
        score_model = DSMScore(
            sizes=[128, 128, 128, 128],
            net_cls=SpectralNormDNN,
            seed=seed,
            activation=nn.ELU(),
            noise_lvl=0.01,
            feature_norm=False,
            batch_norm=False,
        )

    solver_kwargs_ = {} if solver_kwargs is None else dict(solver_kwargs)
    solver_kwargs_.setdefault("verbose", False)

    return ScoreModel(
        model=score_model,
        solver=solver,
        solver_kwargs=solver_kwargs_,
        device=device,
    )


def make_flow_estimator(
    flow_model=None,
    growth_model=None,
    solver="pfm",
    solver_kwargs=None,
    device="cpu",
    score=None,
    seed=0,
):
    """Create a configured ``FlowModel``.

    Parameters
    ----------
    flow_model : torch.nn.Module or None, default=None
        Flow model. If ``None``, a default ``CLEFlow`` is created.
    growth_model : torch.nn.Module or None, default=None
        Optional growth model.
    solver : str, default='pfm'
        Flow solver name.
    solver_kwargs : dict or None, default=None
        Extra kwargs passed to the flow solver.
        The flow solver reads optimization settings from ``solver_kwargs``
        (e.g. ``n_epochs``, ``lr``).
    device : str or torch.device, default='cpu'
        Device used by the flow estimator.
    score : torch.nn.Module or None, default=None
        Score model attached to the flow model.
    seed : int, default=0
        Random seed used when the default flow model is built.
    """
    if flow_model is None:
        flow_model = CLEFlow(
            score=score,
            sizes=[128, 128, 128],
            seed=seed,
            activation=nn.ELU(),
            lx=1.5,
            feature_norm=False,
            full=True,
            batch_norm=False,
        )
    elif score is not None and hasattr(flow_model, "score"):
        flow_model.score = score

    solver_kwargs_ = {} if solver_kwargs is None else dict(solver_kwargs)
    solver_kwargs_.setdefault("verbose", False)

    return FlowModel(
        flow=flow_model,
        growth=growth_model,
        solver=solver,
        solver_kwargs=solver_kwargs_,
        device=device,
    )


def make_pfi_estimator(
    score_model=None,
    flow_model=None,
    growth_model=None,
    s_solver="dsm",
    f_solver="pfm",
    s_solver_kwargs=None,
    f_solver_kwargs=None,
    fit_on_score_samples=False,
    device="cpu",
    seed=0,
):
    """Create a configured composite ``PFI`` estimator."""
    score_estimator = make_score_estimator(
        score_model=score_model,
        solver=s_solver,
        solver_kwargs=s_solver_kwargs,
        device=device,
        seed=seed,
    )
    flow_estimator = make_flow_estimator(
        flow_model=flow_model,
        growth_model=growth_model,
        solver=f_solver,
        solver_kwargs=f_solver_kwargs,
        device=device,
        score=None,
        seed=seed,
    )
    return PFI(
        score_estimator=score_estimator,
        flow_estimator=flow_estimator,
        fit_on_score_samples=bool(fit_on_score_samples),
    )


def hyperopt_pfi(*args, **kwargs):
    """Hyperparameter optimization helper (not implemented in this API version)."""
    _ = args, kwargs
    raise NotImplementedError("hyperopt_pfi is not implemented yet due to changes in the API.")
