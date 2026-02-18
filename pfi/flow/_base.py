"""Base flow estimator API."""

import numpy as np
import torch

from ..utils.data import snapshots_from_X
from ._fm import FM_


class FlowModel:
    """Standard flow estimator trained with flow matching.

    Parameters
    ----------
    flow : torch.nn.Module
        Flow model consuming ``(batch_size, ndim + 1)`` inputs.
    growth : torch.nn.Module or None, default=None
        Optional growth model.
    solver : str, default='fm'
        Solver backend.
    solver_kwargs : dict or None, default=None
        Extra keyword arguments for solver. For ``solver='fm'``, this must
        include ``interp``.
    device : str or torch.device, default='cpu'
        Device used for training and inference.
    """

    def __init__(self, flow=None, growth=None, solver="fm", solver_kwargs=None, device="cpu"):
        self.flow = flow
        self.growth = growth
        self.solver = solver
        self.solver_kwargs = {} if solver_kwargs is None else solver_kwargs
        self.device = device

    def fit(self, X, y=None):
        """Fit drift (and optional growth) from time-augmented samples."""
        dist, times = snapshots_from_X(X)

        self.Ndim_ = X.shape[1] - 1
        self.flow_ = self.flow.to(self.device)
        self.growth_ = self.growth
        if self.growth_ is not None:
            self.growth_ = self.growth_.to(self.device)

        if self.solver == "fm":
            solver_kwargs = dict(self.solver_kwargs)
            interp = solver_kwargs.pop("interp")
            self.flow_, self.growth_, loss_hist = FM_(
                dist,
                times,
                interp,
                self.flow_,
                growth_model=self.growth_,
                device=self.device,
                **solver_kwargs,
            )
            self.loss_ = np.asarray(loss_hist)
        else:
            raise NotImplementedError("No other flow solvers implemented")

        self.times_ = np.unique(X[:, -1])
        self.flow_ = self.flow_.eval()
        return self

    def _predict(self, X, stoch=False):
        """Internal torch prediction hook used by base methods."""
        return self.flow_(X, stoch=stoch)

    def predict(self, X, stoch=False):
        """Predict flow vectors for input states."""
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self._predict(Xt, stoch=stoch)
        return out.detach().cpu().numpy()

    def sample(self, X, Dt, dt=0.01, stoch=False):
        """Simulate trajectories from initial states over ``Dt``."""
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        x = Xt[:, : self.Ndim_].clone()
        t = Xt[:, -1].clone()

        n_steps = int(Dt / dt)
        sqrt_dt = np.sqrt(dt)

        with torch.no_grad():
            for _ in range(n_steps):
                inp = torch.cat([x, t[:, None]], dim=1)
                if stoch:
                    drift, noise = self._predict(inp, stoch=True)
                    x = x + drift * dt + noise * sqrt_dt
                else:
                    drift = self._predict(inp, stoch=False)
                    x = x + drift * dt
                t = t + dt

        return x.detach().cpu().numpy()

    def score(self, X, y, stoch=False, dt=0.01):
        """Compute per-time energy distance between simulated and targets."""
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
