"""Score-matching estimator interface and evaluation utilities."""

import numpy as np
import torch

from sklearn.base import BaseEstimator
from sklearn.base import _fit_context as fit_context
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn_compat.utils.validation import validate_data
from .solvers import DSM_, generate_data_DSM
from ..utils.data import snapshots_from_X, check_random_state_torch
from ..utils.nns import clone_module
from .models import DSMScore

class ScoreEstimator(BaseEstimator):
    """Estimate score functions from snapshot data.

    Parameters
    ----------
    model : torch.nn.Module
        Score model (dimensions of input change depending on solver)
    solver : {'dsm'}, default='dsm'
        Solver backend.
    solver_kwargs : dict, default=None
        Extra keyword arguments passed to the selected solver.
        For ``solver='dsm'``, this can include ``scheduler_kwargs`` to
        configure the internal ``MultiStepLR``.
    device : str or torch.device, default='cpu'
        Device used for training and inference.

    Attributes
    ----------
    Ndim_ : int
        Inferred state dimension, set during `fit`.
    times_ : ndarray of shape (n_times,)
        Sorted unique training times, set during `fit`.
    model_ : torch.nn.Module
        Fitted score model used at inference time, set during `fit`.
        The input of this fitted model is ``(x, sigma, t)``,
        dimension ``ndim + 2``.
    loss_ : ndarray of shape (n_epochs,)
        Recorded training loss history.
    min_noise_ : float
        Minimum noise level used by the fitted DSM model.

    """

    _parameter_constraints = {
        "model": [HasMethods(["to"]), None],
        "solver": [StrOptions({"dsm"})],
        "solver_kwargs": [dict, None],
        "device": [object],
        "random_state": [int, None],
    }

    def __init__(
        self,
        model=None,
        solver="dsm",
        solver_kwargs=None,
        device="cpu",
        random_state=0,
    ):
        self.model = model
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.device = device
        self.random_state = random_state

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
        return hasattr(self, "model_")

    @fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y=None,
    ):
        """Fit the score estimator on time-augmented data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ndim + 1)
            Input data where the last column contains time.
        y : None, default=None
            Ignored. Present for future compatibility with sklearn.

        Returns
        -------
        self : ScoreEstimator
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
        if self.model is None:
            model = DSMScore()
        else:
            model = clone_module(self.model)
        model = model.to(self.device)
        self.times_ = np.unique(X[:, -1])
        solver_kwargs = {} if self.solver_kwargs is None else dict(self.solver_kwargs)

        if self.solver == "dsm":
            self.model_, loss_hist, min_noise = DSM_(
                dist,
                times,
                model,
                generator=generator,
                **solver_kwargs,
            )
            self.loss_ = np.asarray(loss_hist)
            self.min_noise_ = min_noise
            self.model_.noise_lvl = max(self.min_noise_, self.model_.noise_lvl)
        else:
            raise NotImplementedError("Other score matching solvers not implemented yet.")

        self.model_ = self.model_.eval()
        return self

    def predict(
        self,
        X,
    ):
        """Predict score vectors for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, ndim + 1)
            Inputs containing state and time columns.

        Returns
        -------
        score : ndarray of shape (n_samples, ndim)
            Predicted score vectors.
        """
        check_is_fitted(self, "model_")
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype="numeric")
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            score = self.model_(X)

        return score.detach().cpu().numpy()

    def sample(
        self,
        X,
        **kwargs,
    ):
        """Generate samples using the fitted score model.

        Parameters
        ----------
        X : ndarray of shape of shape (n_samples, ndim + 1)
            Conditioning samples with time in the last column.
        **kwargs : dict
            Extra keyword arguments forwarded to the sampler.

        Returns
        -------
        gen : ndarray of shape (nsamples, ndim + 1)
            Generated samples with time in the last column.
        """
        check_is_fitted(self, ["model_", "Ndim_"])
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype="numeric")
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        if self.solver == "dsm":
            nsamples = X.shape[0]
            init_ = torch.zeros(
                (nsamples, self.Ndim_ + 2),
                dtype=torch.float32,
                device=self.device,
            )
            init_[:, 0 : self.Ndim_] = X[:, 0 : self.Ndim_] + 0.1 * torch.randn(
                (nsamples, self.Ndim_),
                dtype=torch.float32,
                device=self.device,
            )
            time_ = X[:, -1]
            with torch.no_grad():
                gen = generate_data_DSM(
                    infNet=self.model_,
                    init_=init_,
                    time_=time_,
                    **kwargs,
                )
            gen_xt = torch.cat(
                [gen[:, : self.Ndim_], gen[:, self.Ndim_ + 1 : self.Ndim_ + 2]],
                dim=1,
            )
            return gen_xt.detach().cpu().numpy()

        raise NotImplementedError("Other score matching solvers not implemented yet.")

    def error(
        self,
        X,
        y=None,
        **kwargs,
    ):
        """Compute per-time energy distance between generated and observed data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ndim + 1)
            Input data with time in the last column.
        y : None, default=None
            Ignored. Present for future compatibility with sklearn.
        **kwargs : dict
            Extra keyword arguments forwarded to the sampler.

        Returns
        -------
        scores : ndarray of shape (n_times,)
            Energy distance at each unique time in ``X``.
        """
        _ = y
        import geomloss

        check_is_fitted(self, ["model_", "Ndim_"])
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype="numeric")
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        times = torch.unique(X[:, -1])
        scores = []

        loss = geomloss.SamplesLoss("energy")
        with torch.no_grad():
            for t in times:
                x_t = X[X[:, -1] == t]
                gen = self.sample(
                    x_t.cpu().numpy(),
                    **kwargs,
                )
                y_t = x_t[:, : self.Ndim_].contiguous()
                ed = loss(
                    torch.as_tensor(gen[:, : self.Ndim_], dtype=torch.float32, device=self.device).contiguous(),
                    y_t,
                ).item()
                scores.append(ed)

        return np.asarray(scores)

    def score(
        self,
        X,
        y=None,
        **kwargs,
    ):
        """Return the negated mean error for sklearn compatibility. 
        This defines a proper score that can be used for cross-validation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ndim + 1)
            Input data with time in the last column.
        y : None, default=None
            Ignored. Present for future compatibility with sklearn.
        **kwargs : dict
            Extra keyword arguments forwarded to ``error``.

        Returns
        -------
        float
            Negated mean error. Larger values indicate better fit.
        """
        errors = self.error(X, y=y, **kwargs)
        if errors.size == 0:
            return float("nan")
        return -float(np.mean(errors))
