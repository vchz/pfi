import numpy as np
import torch
import torch.nn as nn

from pfi.score import ScoreEstimator
from pfi.score.models import DSMScore
from pfi.utils.data import X_from_snapshots
from pfi.utils.simulations import simulate_ornstein_uhlenbeck

def test_dsm_OU():
    """Fit DSM on OU data and check time-wise sample quality."""
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
    score_model = DSMScore(
        sizes=[Np, Np, Np, Np, Np],
        spectral_norm=False,
        activation=nn.ELU(),
        noise_lvl=0.0,
        batch_norm='zscore'
    ).to(device)

    score_reg = ScoreEstimator(
        model=score_model,
        solver="dsm",
        solver_kwargs={
            'L': 10, 
            'lr': 1e-2, 
            'n_epochs': 5000, 
            'bs': None, 
            'adaptive_weights': 1,
            'verbose': False,
        },
        device=device,
    )
    score_reg.fit(X)

    eds = score_reg.score(X)
    assert np.all(eds < 0.055)
