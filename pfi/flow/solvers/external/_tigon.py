"""TIGON external solver placeholder."""


def TIGON_(dist, times, flow_model, growth_model=None, **solver_kwargs):
    """Placeholder TIGON solver.

    Parameters
    ----------
    dist : list of torch.Tensor
        Snapshot tensors.
    times : torch.Tensor
        Snapshot times.
    flow_model : torch.nn.Module or None
        Unused placeholder for API compatibility.
    growth_model : torch.nn.Module or None, default=None
        Unused placeholder for API compatibility.
    **solver_kwargs : dict
        Unused solver options.

    Raises
    ------
    NotImplementedError
        Always, because TIGON backend is not implemented.
    """
    _ = dist, times, flow_model, growth_model, solver_kwargs
    raise NotImplementedError("TIGON solver is not implemented yet.")
