import numpy as np
import torch
import torch.nn as nn

from pfi.score import ScoreMatching
from pfi.utils import DNN, X_from_snapshots, simulate_ornstein_uhlenbeck


def test_dsm_score_energy_distance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    Np = 50
    score_model = DNN([ndim + 2, Np, Np, Np, Np, Np, ndim], activation=nn.ELU()).to(device)

    score_reg = ScoreMatching(
        model=score_model,
        solver="dsm",
        solver_kwargs=dict(L=10, lr=1e-2, n_epochs=5000, bs=None, adp_flag=1),
        device=device,
    )
    score_reg.fit(X)

    eds = score_reg.score(X)
    assert np.all(eds < 0.03)
