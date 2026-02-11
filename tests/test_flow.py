import numpy as np
import torch

from pfi.flow import FlowRegression
from pfi.flow.interpolants import ChebyshevInterpolant
from pfi.flow.models import OUFlow
from pfi.score.models import OUScore
from pfi.utils.data import X_from_snapshots
from pfi.utils.simulations import simulate_ornstein_uhlenbeck

def test_fm_chebyshev_recovers_ou_b():
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

    interp = ChebyshevInterpolant(device=device)

    B_ = torch.tensor(B, dtype=torch.float32, device=device)
    D_ = torch.tensor(D * np.eye(ndim), dtype=torch.float32, device=device)
    m0_ = torch.tensor(m0, dtype=torch.float32, device=device)
    S0_ = torch.tensor(np.eye(ndim) * S0, dtype=torch.float32, device=device)

    true_score = OUScore(net=B_, m0=m0_, S0=S0_, D=D_)

    flow_model = OUFlow(
        net=torch.tensor(np.zeros((2, 2)), dtype=torch.float32, device=device, requires_grad=True),
        score=true_score,
        D=D_,
    )

    flow_reg = FlowRegression(
        interp=interp,
        model=flow_model,
        growth_model=None,
        solver="fm",
        solver_kwargs=dict(n_epochs=5000, lr=1e-2),
        device=device,
    )
    flow_reg.fit(X)

    B_inf = flow_reg.model_.B.detach().cpu().numpy()
    rel_err = np.linalg.norm(B_inf - B) / np.linalg.norm(B)
    assert rel_err < 0.07
