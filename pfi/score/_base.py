"""Score-matching estimator interface and evaluation utilities."""

import numpy as np
import torch

from .solvers import DSM_, generate_data_DSM
from ..utils.data import snapshots_from_X

class ScoreEstimator:
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

    def __init__(
        self,
        model,
        solver="dsm",
        solver_kwargs=None,
        device="cpu",
    ):
        self.model = model
        self.solver = solver
        self.solver_kwargs = solver_kwargs if solver_kwargs is not None else {}
        self.device = device

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
        dist, times = snapshots_from_X(X)
        dist = [d.to(device=self.device, dtype=torch.float32) for d in dist]
        times = times.to(device=self.device, dtype=torch.float32)

        self.Ndim_ = X.shape[1] - 1
        self.model = self.model.to(self.device)
        self.times_ = np.unique(X[:, -1])

        if self.solver == "dsm":
            self.model_, loss_hist, min_noise = DSM_(
                dist,
                times,
                self.model,
                **self.solver_kwargs,
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

    def score(
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

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        times = torch.unique(X[:, -1])
        scores = []

        loss = geomloss.SamplesLoss("energy")
        with torch.no_grad():
            for t in times:
                x_t = X[X[:, -1] == t]
                gen = self.sample(
                    x_t,
                    **kwargs,
                )
                y_t = x_t[:, : self.Ndim_].contiguous()
                ed = loss(
                    torch.as_tensor(gen[:, : self.Ndim_], dtype=torch.float32, device=self.device).contiguous(),
                    y_t,
                ).item()
                scores.append(ed)

        return np.asarray(scores)
