import numpy as np
import torch

from pfi.flow.interpolants import ChebyshevInterpolant, MMOT_trajectories
from pfi.flow.models import OUFlow
from pfi.score.models import OUScore
from pfi.utils import X_from_snapshots, simulate_ornstein_uhlenbeck


def test_chebyshev_interpolant_matches_ou_flow():
    dt = 0.01
    Dt = 0.2
    nsnaps = 5
    nsamples = 2000
    D = 5.0
    ndim = 2

    np.random.seed(7)
    torch.manual_seed(7)

    A = np.random.randn(2, 2)
    B = A @ A.T

    m0 = 80 * np.ones((ndim), dtype=np.float32)
    S0 = 8.0

    samples_full, tt = simulate_ornstein_uhlenbeck(
        Om=B,
        D=D * np.eye(ndim),
        m0=m0,
        S0=S0,
        nsamples=nsamples,
        ndim=ndim,
        Dt=Dt,
        K=nsnaps,
        dt=dt,
    )

    X = X_from_snapshots(samples_full, tt)

    np.random.seed(0)
    dist, batch_ot_samples = MMOT_trajectories([torch.tensor(s, dtype=torch.float32) for s in samples_full])

    npoints = 50
    nodes_fit = torch.tensor(tt, dtype=torch.float32)[None, :].repeat(nsamples, 1)
    nodes_eval = torch.tensor(np.linspace(tt[0], tt[-1], npoints), dtype=torch.float32)[None, :].repeat(nsamples, 1)

    batch_ot_samples = torch.tensor(batch_ot_samples, dtype=torch.float32)
    dist = torch.permute(batch_ot_samples[:, :, 0:ndim], (1, 0, 2))

    interp = ChebyshevInterpolant()

    B_ = torch.tensor(B, dtype=torch.float32)
    m0_ = torch.tensor(m0, dtype=torch.float32)
    S0_ = torch.tensor(np.eye(ndim) * S0, dtype=torch.float32)
    D_ = torch.tensor(np.eye(ndim) * D, dtype=torch.float32)

    true_score = OUScore(B=B_, m0=m0_, S0=S0_, D=D_)
    true_flow = OUFlow(B=B_, score=true_score, D=D_)

    dx_flow = (
        true_flow(torch.tensor(X, dtype=torch.float32))
        .reshape(nsnaps, nsamples, ndim)
        .permute(1, 0, 2)
        .detach()
        .numpy()
    )

    interp.fit(nodes_fit, dist)
    x_interp, _ = interp.predict(nodes_eval)
    _, dx_interp_data = interp.predict(nodes_fit)

    x_interp = x_interp.numpy()
    dx_interp_data = dx_interp_data.numpy()

    average_rel_err = np.mean(
        ((dx_flow - dx_interp_data) ** 2).sum(axis=-1) / ((dx_flow) ** 2).sum(axis=-1),
        axis=0,
    )
    assert np.all(average_rel_err[:-1] < 0.007)
