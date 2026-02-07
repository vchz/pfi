"""Flow regression estimator wrapping flow-matching solvers."""

import numpy as np
import torch

from .solvers._fm import FM_
from ..utils.data import snapshots_from_X


class FlowRegression:
    """Estimate drift (and optional growth) models from snapshot data.

    Parameters
    ----------
    interp : object
        Interpolant object implementing ``fit`` and ``predict``.
    model : torch.nn.Module
        Drift model consuming inputs of shape ``(batch_size, ndim + 1)``.
    growth_model : torch.nn.Module or None, default=None
        Optional growth model used for unbalanced transport.
    dt : float, default=1.0
        Time step used by `predict` when advancing one step.
    solver : {'fm'}, default='fm'
        Solver backend.
    solver_kwargs : dict, default=None
        Extra keyword arguments passed to the selected solver.
    device : str or torch.device, default='cpu'
        Device used for training and inference.

    Attributes
    ----------
    Ndim_ : int
        Inferred state dimension, set during `fit`.
    model_ : torch.nn.Module
        Fitted drift model, set during `fit`.
    growth_model_ : torch.nn.Module or None
        Fitted growth model when provided, set during `fit`.
    times_ : ndarray of shape (n_times,)
        Sorted unique training times, set during `fit`.
    """

    def __init__(
        self,
        interp,
        model,
        growth_model=None,
        dt=1.0,
        solver="fm",
        solver_kwargs=None,
        device="cpu",
    ):
        self.model = model
        self.interp = interp
        self.growth_model = growth_model
        self.dt = dt
        self.solver = solver
        self.solver_kwargs = solver_kwargs if solver_kwargs is not None else {}
        self.device = device

    def fit(
        self,
        X,
        y=None,
    ):
        """Fit flow models from time-augmented samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ndim + 1)
            Input data where the last column contains time.
        y : None, default=None
            Ignored. Present for estimator API compatibility.

        Returns
        -------
        self : FlowRegression
            Fitted estimator.
        """
        dist, times = snapshots_from_X(X)

        self.Ndim_ = X.shape[1] - 1
        self.model_ = self.model.to(self.device)
        self.growth_model_ = self.growth_model
        if self.growth_model_ is not None:
            self.growth_model_ = self.growth_model_.to(self.device)

        if self.solver == "fm":
            self.model_, self.growth_model_ = FM_(
                dist,
                times,
                self.interp,
                self.model_,
                growth_model=self.growth_model_,
                device=self.device,
                **self.solver_kwargs,
            )
        else:
            raise NotImplementedError("Other flow regression solvers (sde, ode, cnf) not implemented")

        self.times_ = np.unique(X[:, -1])
        return self

    def predict(
        self,
        X,
    ):
        """Predict next-state positions by Euler stepping the learned drift.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Input states with time in the last column.

        Returns
        -------
        x_next : ndarray of shape (n_samples, ndim)
            One-step predictions ``x + dt * f(x, t)``.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        x = X[:, : self.Ndim_]
        t = X[:, -1:]
        inp = torch.cat([x, t], dim=1)
        drift = self.model_(inp)
        x_next = x + self.dt * drift
        return x_next.detach().cpu().numpy()

    def score(
        self,
        X,
        y,
    ):
        """Compute per-time energy distance between predictions and targets.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ndim + 1)
            Inputs used for prediction.
        y : ndarray of shape (n_targets, ndim) or (n_targets, ndim + 1)
            Targets. If time is present, rows at ``t + dt`` are selected.

        Returns
        -------
        scores : ndarray of shape (n_times,)
            Energy distance for each unique input time.
        """
        import geomloss

        X = np.asarray(X)
        y = np.asarray(y)
        times = np.unique(X[:, -1])
        scores = []

        loss = geomloss.SamplesLoss("energy")
        for t in times:
            x_t = X[X[:, -1] == t]
            pred = self.predict(x_t)
            if y.shape[1] == self.Ndim_ + 1:
                y_t = y[np.isclose(y[:, -1], t + self.dt)]
                y_t = y_t[:, : self.Ndim_]
            else:
                y_t = y
            ed = loss(
                torch.tensor(pred, dtype=torch.float32, device=self.device),
                torch.tensor(y_t, dtype=torch.float32, device=self.device),
            ).item()
            scores.append(ed)

        return np.asarray(scores)
