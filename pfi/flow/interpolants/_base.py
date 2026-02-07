"""Base interpolant interface used by flow-matching solvers."""


class BaseInterpolant:
    """Abstract base class for trajectory interpolants.

    Parameters
    ----------
    device : str or torch.device, default='cpu'
        Device used by derived interpolants.

    """

    def __init__(
        self,
        device="cpu",
    ):
        self.device = device

    def fit(
        self,
        nodes_fit,
        Dist,
    ):
        """Store fit data and call implementation-specific fitting.

        Parameters
        ----------
        nodes_fit : torch.Tensor of shape (batch_size, n_nodes)
            Time nodes used for fitting.
        Dist : torch.Tensor of shape (batch_size, n_nodes, ndim)
            Trajectory values at ``nodes_fit``.

        Returns
        -------
        self : BaseInterpolant
            Fitted interpolant.
        """
        self.y_fit_ = Dist
        self.t_fit_ = nodes_fit

        self._fit()

        return self

    def predict(self, t_eval):
        """Evaluate interpolant and derivative at requested nodes.

        Parameters
        ----------
        t_eval : torch.Tensor of shape (batch_size, n_eval)
            Evaluation nodes.

        Returns
        -------
        x_interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Interpolated trajectories.
        dx_interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Time derivatives at evaluation nodes.
        """
        x_interp, dx_interp = self._predict(t_eval)
        return x_interp, dx_interp

    def _predict(self, t_eval):
        """Implementation hook for prediction.

        Parameters
        ----------
        t_eval : torch.Tensor of shape (batch_size, n_eval)
            Evaluation nodes.

        Returns
        -------
        x_interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Interpolated trajectories.
        dx_interp : torch.Tensor of shape (batch_size, n_eval, ndim)
            Time derivatives.
        """
        raise NotImplementedError

    def _fit(
        self,
    ):
        """Implementation hook for fitting internal interpolant state.

        Returns
        -------
        self : BaseInterpolant
            Fitted interpolant.
        """
        raise NotImplementedError
