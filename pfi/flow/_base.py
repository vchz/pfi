"""Base flow estimator API."""

import numpy as np
import torch

from ..utils.data import snapshots_from_X
from .solvers import PFM_, UPFI_
from .solvers.future import UFM_UOT_, UFM_OT_
from .solvers.external import DeepRUOTv2_


SOLVER_FUNCS = {
    "pfm": PFM_,
    "upfi": UPFI_,
    "external.deepruotv2": DeepRUOTv2_,
    "future.ufm_uot": UFM_UOT_,
    "future.ufm_ot": UFM_OT_,
}


class FlowModel:
    """Flow estimator trained with one of the configured solver backends.

    Supported solvers are configured in ``SOLVER_FUNCS``.
    """

    def __init__(self, flow=None, growth=None, solver="pfm", solver_kwargs=None, device="cpu"):
        """Initialize a flow estimator.

        Parameters
        ----------
        flow : torch.nn.Module, optional
            Drift model. Ignored if the solver is an external solver.
        growth : torch.nn.Module, optional
            Growth model used by unbalanced solvers. Ignored if the solver is an external solver.
        solver : str, default="pfm"
            Solver name. Must be a key of ``SOLVER_FUNCS``.
        solver_kwargs : dict, optional
            Extra solver parameters forwarded to the selected solver.
        device : str, default="cpu"
            Torch device used for fitting and inference.
        """
        self.solver = solver
        self.solver_kwargs = {} if solver_kwargs is None else solver_kwargs
        self.device = device

        self.flow = flow.to(self.device) if flow is not None else None
        self.growth = growth.to(self.device) if growth is not None else None

    def fit(self, X, y=None):
        """Fit flow parameters from time-augmented samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Training data with time in the last column.
        y : None, optional
            Unused, kept for estimator API compatibility.

        Returns
        -------
        self : FlowModel
            Fitted estimator.
        """
        _ = y
        dist, times = snapshots_from_X(X)
        dist = [d.to(device=self.device, dtype=torch.float32) for d in dist]
        times = times.to(device=self.device, dtype=torch.float32)

        self.Ndim_ = X.shape[1] - 1
        solver_func = SOLVER_FUNCS.get(self.solver)
        if solver_func is None:
            raise ValueError(f"Unknown solver '{self.solver}'.")

        self.flow_, self.growth_, loss_hist = solver_func(
            dist,
            times,
            self.flow,
            growth_model=self.growth,
            **self.solver_kwargs,
        )

        self.loss_ = np.asarray(loss_hist)
        self.times_ = np.unique(X[:, -1])
        if hasattr(self.flow_, "eval"):
            self.flow_ = self.flow_.eval()
        if hasattr(self.growth_, "eval"):
            self.growth_ = self.growth_.eval()
        return self

    def predict(self, X, stoch=False):
        """Predict flow vectors for input states.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Inputs with time in the last column.
        stoch : bool, default=False
            Whether to request stochastic output.

        Returns
        -------
        y_pred : ndarray
            Predicted drift values.
        """
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.flow_(Xt, stoch=stoch)
        return out.detach().cpu().numpy()

    def sample(self, X, Dt, dt=0.01, stoch=False, pos=True):
        """Simulate trajectories from initial states over ``Dt``.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Initial states with time in the last column.
        Dt : float
            Simulation horizon.
        dt : float, default=0.01
            Integration step.
        stoch : bool, default=False
            Whether to use stochastic model outputs.
        pos : bool, default=True
            If ``True``, applies ``relu`` to keep simulated states nonnegative
            in stochastic mode.

        Returns
        -------
        x_final : ndarray of shape (n_samples, ndim)
            Simulated final states.
        """
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        x = Xt[:, : self.Ndim_].clone()
        t = Xt[:, -1].clone()

        n_steps = int(Dt / dt)
        sqrt_dt = np.sqrt(dt)

        with torch.no_grad():
            for _ in range(n_steps):
                inp = torch.cat([x, t[:, None]], dim=1)
                if stoch:
                    drift, noise = self.flow_(inp, stoch=True)
                    if pos:
                        x = torch.relu(x + drift * dt + noise * sqrt_dt)
                    else:
                        x = x + drift * dt + noise * sqrt_dt
                else:
                    drift = self.flow_(inp, stoch=False)
                    x = x + drift * dt
                t = t + dt

        return x.detach().cpu().numpy()

    def score(self, X, y, stoch=False, dt=0.01, pos=True):
        """Compute per-time energy distance after pushing to next target times.

        Parameters
        ----------
        X : array-like of shape (n_samples_x, ndim + 1)
            Source samples grouped by time in the last column.
        y : array-like of shape (n_samples_y, ndim + 1)
            Target samples grouped by time in the last column.
        stoch : bool, default=False
            Whether to simulate with stochastic dynamics.
        dt : float, default=0.01
            Integration step for sampling.
        pos : bool, default=True
            Passed to :meth:`sample` to enforce nonnegative states in
            stochastic mode.

        Returns
        -------
        scores : ndarray of shape (n_time_pairs,)
            Energy distance for each source time in ``X`` that has a strictly
            later time in ``y``.
        """
        import geomloss

        X = np.asarray(X)
        y = np.asarray(y)
        x_times = np.sort(np.unique(X[:, -1]))
        y_times = np.sort(np.unique(y[:, -1]))
        scores = []

        loss = geomloss.SamplesLoss("energy")
        for tx in x_times:
            y_later = y_times[y_times > tx]
            if y_later.size == 0:
                continue
            ty = y_later[0]
            x_t = X[np.isclose(X[:, -1], tx)]
            y_t = y[np.isclose(y[:, -1], ty)][:, : self.Ndim_]
            pred = self.sample(x_t, Dt=(ty - tx), stoch=stoch, dt=dt, pos=pos)
            ed = loss(
                torch.as_tensor(pred, dtype=torch.float32, device=self.device),
                torch.as_tensor(y_t, dtype=torch.float32, device=self.device),
            ).item()
            scores.append(ed)

        return np.asarray(scores)
