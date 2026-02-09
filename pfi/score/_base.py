"""Score-matching estimator interface and evaluation utilities."""

import numpy as np
import torch

from .solvers._dsm import DSM_, generate_data_DSM
from ..utils.nns import FreezeVarDNN
from ..utils.data import snapshots_from_X


class ScoreMatching:
    """Estimate score functions from snapshot data.

    Parameters
    ----------
    model : torch.nn.Module
        Score model (dimensions of input change depending on solver)
    solver : {'dsm'}, default='dsm'
        Solver backend.
    solver_kwargs : dict, default=None
        Extra keyword arguments passed to the selected solver.
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
        The input of this fitted model is (x,t), dimension ndim + 1.
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
            Ignored. Present for estimator API compatibility.

        Returns
        -------
        self : ScoreMatching
            Fitted estimator.
        """
        dist, times = snapshots_from_X(X)

        self.Ndim_ = X.shape[1] - 1
        self.model = self.model.to(self.device)
        self.times_ = np.unique(X[:, -1])

        if self.solver == "dsm":
            self.model, loss_hist = DSM_(
                dist,
                times,
                self.model,
                device=self.device,
                **self.solver_kwargs,
            )
            self.loss_ = np.asarray(loss_hist)
            self.model_ = FreezeVarDNN(
                dnn=self.model,
                var_index=self.Ndim_,
                var_value=0.01,
            )
            self.model.eval()
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
            Inputs containing state and time columns. The internal noise-level
            feature is inserted by `model_`.

        Returns
        -------
        score : ndarray of shape (n_samples, ndim)
            Predicted score vectors.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        score = self.model_(X)

        return score.detach().cpu().numpy()

    def sample(
        self,
        X,
        nsamples=None,
        maxiter=100,
    ):
        """Generate samples using the fitted score model.

        Parameters
        ----------
        X : ndarray of shape of shape (n_samples, ndim + 1)
            Conditioning samples with time in the last column.
        nsamples : int, default=None
            Number of generated samples. If ``None``, uses ``X.shape[0]``.
        maxiter : int, default=100
            Number of Langevin updates per noise level.

        Returns
        -------
        gen : ndarray of shape (nsamples, ndim)
            Generated samples.
        """
        X = torch.tensor(X, 
                         dtype=torch.float32, 
                         device=self.device)
        
        if nsamples is None:
            nsamples = X.shape[0]

        if self.solver == "dsm":

            init_ = 4*torch.rand((nsamples,self.Ndim_+2)) + 1
            init_[:,0:self.Ndim_] = X[:,0:self.Ndim_]
            time_ = X[0, -1]
            with torch.no_grad():
                gen, _ = generate_data_DSM(
                    maxiter=maxiter,
                    infNet=self.model,
                    nsamples=nsamples,
                    init_=init_,
                    time_=time_,
                    L=self.solver_kwargs["L"],
                    ndim=self.Ndim_,
                    device=self.device,
                )
            return gen[:, : self.Ndim_]

        raise NotImplementedError("Langevin sampling not implemented yet.")

    def score(
        self,
        X,
        y=None,
        maxiter=100,
    ):
        """Compute per-time energy distance between generated and observed data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ndim + 1)
            Input data with time in the last column.
        y : None, default=None
            Ignored. Present for estimator API compatibility.
        maxiter : int, default=100
            Number of Langevin updates used during sampling.

        Returns
        -------
        scores : ndarray of shape (n_times,)
            Energy distance at each unique time in ``X``.
        """
        import geomloss

        X = torch.tensor(X, 
                         dtype=torch.float32, 
                         device=self.device)
        times = torch.unique(X[:, -1])
        scores = []

        loss = geomloss.SamplesLoss("energy")
        for t in times:
            x_t = X[X[:, -1] == t]
            gen = self.sample(x_t, nsamples=x_t.shape[0], maxiter=maxiter)
            y_t = x_t[:, : self.Ndim_]
            ed = loss(
                torch.tensor(gen, dtype=torch.float32, device=self.device),
                torch.tensor(y_t, dtype=torch.float32, device=self.device),
            ).item()
            scores.append(ed)

        return np.asarray(scores)
