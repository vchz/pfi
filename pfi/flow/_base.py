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
    solver : {'fm'}, default='fm'
        Solver backend.
    solver_kwargs : dict, default=None
        Extra keyword arguments passed to the selected solver.
        For ``solver='fm'``, this can include ``scheduler_kwargs`` to
        configure the internal ``MultiStepLR``.
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
        solver="fm",
        solver_kwargs=None,
        device="cpu",
    ):
        self.model = model
        self.interp = interp
        self.growth_model = growth_model
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
            self.model_, self.growth_model_, loss_hist = FM_(
                dist,
                times,
                self.interp,
                self.model_,
                growth_model=self.growth_model_,
                device=self.device,
                **self.solver_kwargs,
            )
            self.loss_ = np.asarray(loss_hist)
        else:
            raise NotImplementedError("Other flow regression solvers (sde, ode, cnf) not implemented")

        self.times_ = np.unique(X[:, -1])
        self.model_ = self.model_.eval()
        return self

    def predict(
        self,
        X,
    ):
        """Predict flow vectors for input states.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Input states with time in the last column.

        Returns
        -------
        drift : ndarray of shape (n_samples, ndim)
            Predicted flow vectors ``f(x, t)``.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            drift = self.model_(X)
        return drift.detach().cpu().numpy()

    def sample(
        self,
        X,
        Dt,
        dt=0.01,
        stoch=False,
    ):
        """Simulate trajectories from initial states over a duration ``Dt``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ndim + 1)
            Initial states with time in the last column.
        Dt : float
            Simulation duration.
        dt : float, default=0.01
            Euler integration step.
        stoch : bool, default=False
            If ``True``, use stochastic simulation with model-provided
            noise terms. If ``False``, use deterministic Euler flow.

        Returns
        -------
        x_final : ndarray of shape (n_samples, ndim)
            Simulated states after duration ``Dt``.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        x = X[:, : self.Ndim_].clone()
        t = X[:, -1].clone()

        n_steps = int(Dt / dt)
        sqrt_dt = np.sqrt(dt)

        with torch.no_grad():
            for _ in range(n_steps):
                inp = torch.cat([x, t[:, None]], dim=1)
                if stoch:
                    drift, noise = self.model_(inp, stoch=True)
                    x = x + drift * dt + noise * sqrt_dt
                else:
                    drift = self.model_(inp, stoch=False)
                    x = x + drift * dt
                t = t + dt

        return x.detach().cpu().numpy()

    def score(
        self,
        X,
        y,
        stoch=False,
        dt=0.01,
    ):
        """Compute per-time energy distance between simulated and target data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ndim + 1)
            Source samples with time in the last column.
        y : ndarray of shape (n_targets, ndim + 1)
            Target samples with time in the last column.
        stoch : bool, default=False
            If ``True``, use stochastic simulation in ``sample``.

        Returns
        -------
        scores : ndarray of shape (n_pairs,)
            Energy distance for each paired source/target time.
        """
        import geomloss

        X = np.asarray(X)
        y = np.asarray(y)
        x_times = np.sort(np.unique(X[:, -1]))
        y_times = np.sort(np.unique(y[:, -1]))
        npairs = min(len(x_times), len(y_times))
        scores = []

        loss = geomloss.SamplesLoss("energy")
        for i in range(npairs):
            tx = x_times[i]
            ty = y_times[i]
            x_t = X[np.isclose(X[:, -1], tx)]
            y_t = y[np.isclose(y[:, -1], ty)][:, : self.Ndim_]
            pred = self.sample(x_t, Dt=(ty - tx), stoch=stoch, dt=dt)
            ed = loss(
                torch.tensor(pred, dtype=torch.float32, device=self.device),
                torch.tensor(y_t, dtype=torch.float32, device=self.device),
            ).item()
            scores.append(ed)

        return np.asarray(scores)
