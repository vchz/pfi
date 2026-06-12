"""Base flow estimator API."""

import numpy as np
import torch

from sklearn.base import BaseEstimator
from sklearn.base import _fit_context as fit_context
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn_compat.utils.validation import validate_data
from ..utils.data import snapshots_from_X, check_random_state_torch
from ..utils.nns import clone_module
from .solvers import PFM_, UPFI_
from .solvers.future import UFM_UOT_, UFM_OT_
from .solvers.external import DeepRUOTv2_
from .models import AutonomousODEFlow

SOLVER_FUNCS = {
    "pfm": PFM_,
    "upfi": UPFI_,
    "external.deepruotv2": DeepRUOTv2_,
    "future.ufm_uot": UFM_UOT_,
    "future.ufm_ot": UFM_OT_,
}

class FlowEstimator(BaseEstimator):
    """Flow estimator trained with one of the configured solver backends.

    Supported solver names are given by ``SOLVER_FUNCS``.

    Parameters
    ----------
    flow : torch.nn.Module or None, default=None
        Drift model.
    growth : torch.nn.Module or None, default=None
        Optional growth model used by unbalanced solvers.
    solver : str, default='pfm'
        Solver backend name.
    solver_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to the selected solver.
    device : str or torch.device, default='cpu'
        Device used for fitting and inference.

    Attributes
    ----------
    Ndim_ : int
        Inferred spatial dimension set during ``fit``.
    flow_ : torch.nn.Module
        Fitted drift model returned by the selected solver.
    growth_ : torch.nn.Module or None
        Fitted growth model returned by the selected solver.
    loss_ : ndarray of shape (n_epochs,)
        Recorded training loss history.
    times_ : ndarray of shape (n_times,)
        Sorted unique training times.

    """

    _parameter_constraints = {
        "flow": [HasMethods(["to"]), None],
        "growth": [HasMethods(["to"]), None],
        "solver": [StrOptions(set(SOLVER_FUNCS))],
        "solver_kwargs": [dict, None],
        "device": [object],
        "random_state": [int, None],
    }

    def __init__(self, flow=None, growth=None, solver="pfm", solver_kwargs=None, device="cpu", random_state=0):

        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.device = device
        self.random_state = random_state

        self.flow = flow
        self.growth = growth

    def _more_tags(self):
        """Return legacy sklearn tags for this estimator."""
        return {"non_deterministic": True, "multioutput_only": True}

    def __sklearn_tags__(self):
        """Set sklearn ≥ 1.6 tags (non-deterministic, multi-output)."""
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.target_tags.multi_output = True
        return tags

    def __sklearn_is_fitted__(self):
        """Return whether the estimator has been fitted."""
        return hasattr(self, "flow_")

    @fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit flow parameters from time-augmented samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Training data with time in the last column.
        y : None, optional
            Ignored. Present for future compatibility with sklearn.

        Returns
        -------
        self : FlowEstimator
            Fitted estimator.
        """
        _ = y

        generator = check_random_state_torch(self.random_state, device=self.device)

        X = validate_data(self, X, reset=True, ensure_2d=True, dtype="numeric")
        if X.shape[1] < 2:
            raise ValueError(
                f"X has 1 feature(s), but {type(self).__name__} requires at least 2 "
                "(1 spatial + 1 time column)."
            )
        dist, times = snapshots_from_X(X)
        if len(dist) < 2:
            raise ValueError(
                "n_samples=1 is not enough. Requires at least 2 distinct time points."
            )
        dist = [d.to(device=self.device, dtype=torch.float32) for d in dist]
        times = times.to(device=self.device, dtype=torch.float32)

        self.Ndim_ = X.shape[1] - 1
        solver_func = SOLVER_FUNCS.get(self.solver)
        if solver_func is None:
            raise ValueError(f"Unknown solver '{self.solver}'.")

        if self.flow is None:
            flow = AutonomousODEFlow().to(self.device)
        else:
            flow = clone_module(self.flow).to(self.device)
        growth = clone_module(self.growth).to(self.device) if self.growth is not None else None

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        if hasattr(flow, "init_model"):
            flow.init_model(X_tensor, generator=generator)
        if growth is not None and hasattr(growth, "init_model"):
            growth.init_model(X_tensor, generator=generator)

        solver_kwargs = {} if self.solver_kwargs is None else dict(self.solver_kwargs)

        self.flow_, self.growth_, loss_hist = solver_func(
            dist,
            times,
            flow,
            growth_model=growth,
            generator=generator,
            **solver_kwargs,
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
        check_is_fitted(self, ["flow_", "Ndim_"])
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype="numeric")
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
        check_is_fitted(self, ["flow_", "Ndim_"])
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype="numeric")
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

    def error(self, X, y=None, stoch=False, dt=0.01, pos=True):
        """Compute per-time energy distance over consecutive snapshot pairs.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Samples from all time points with time in the last column.
        y : ignored
            Present for sklearn API compatibility.
        stoch : bool, default=False
            Whether to simulate with stochastic dynamics.
        dt : float, default=0.01
            Integration step for sampling.
        pos : bool, default=True
            Passed to ``sample`` to enforce nonnegative states in
            stochastic mode.

        Returns
        -------
        scores : ndarray of shape (n_times - 1,)
            Energy distance for each consecutive pair ``(t_k, t_{k+1})``.
        """
        import geomloss

        check_is_fitted(self, ["flow_", "Ndim_"])
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype="numeric")

        _, times = snapshots_from_X(X)
        times = times.numpy()

        scores = []
        loss = geomloss.SamplesLoss("energy")
        for k in range(len(times) - 1):
            tx, ty = float(times[k]), float(times[k + 1])
            x_src = X[np.isclose(X[:, -1], tx)]
            y_tgt = X[np.isclose(X[:, -1], ty)][:, : self.Ndim_]
            pred = self.sample(x_src, Dt=(ty - tx), stoch=stoch, dt=dt, pos=pos)
            ed = loss(
                torch.as_tensor(pred, dtype=torch.float32, device=self.device),
                torch.as_tensor(y_tgt, dtype=torch.float32, device=self.device),
            ).item()
            scores.append(ed)

        return np.asarray(scores)

    def score(self, X, y=None, stoch=False, dt=0.01, pos=True):
        """Return the negated mean error for sklearn compatibility.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Samples from all time points with time in the last column.
        y : ignored
            Present for sklearn API compatibility.
        stoch : bool, default=False
            Whether to simulate with stochastic dynamics.
        dt : float, default=0.01
            Integration step for sampling.
        pos : bool, default=True
            Passed to ``sample`` to enforce nonnegative states in
            stochastic mode.

        Returns
        -------
        float
            Negated mean error over consecutive snapshot pairs. Larger values indicate better fit.
        """
        errors = self.error(X, stoch=stoch, dt=dt, pos=pos)
        if errors.size == 0:
            return float("nan")
        return -float(np.mean(errors))
