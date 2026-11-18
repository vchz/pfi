import numpy as np
import torch

from pfi.flow.interpolants import ChebyshevInterpolant, LinearInterpolant
from pfi.flow.models import AutonomousODEFlow, OUFlow
from pfi.flow.paths import (
    DebiasedSinkhorn,
    ExactOT,
    GaussianDebiasedSinkhorn,
    GaussianOT,
    Independent,
    Sinkhorn,
)
from pfi.flow.solvers import PFM_, PFM_LINEAR_
from pfi.score.models import OUScore
from pfi.utils.nns import ConstantNN
from pfi.utils.simulations import (
    ou_moments,
    ou_pfm_limit,
    random_ou,
    sample_ornstein_uhlenbeck,
)


def _snapshots(sizes, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return [
        torch.randn(n, 3, generator=gen, dtype=dtype) * (1 + 0.3 * k) + k
        for k, n in enumerate(sizes)
    ]


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_path_shapes_and_push():
    """Every ensemble returns paths and masses, and an identity push changes nothing."""
    dist = _snapshots([40, 40, 40])
    for ensemble in [Independent(), ExactOT(), Sinkhorn(reg=1.0), DebiasedSinkhorn(reg=1.0)]:
        paths, masses = ensemble.fit(dist).sample()
        assert paths.shape[1:] == (3, 3)
        assert masses.shape == paths.shape[:2]
        assert torch.isfinite(paths).all()

    gen = torch.Generator().manual_seed(4)
    a = Sinkhorn(reg=0.5).fit(dist, generator=gen).sample(generator=gen)
    gen = torch.Generator().manual_seed(4)
    b = Sinkhorn(reg=0.5).fit(dist, push=lambda x, k: x, generator=gen).sample(generator=gen)
    assert _same(a, b)


def test_gaussian_paths_on_laws():
    """On exact laws, Gaussian OT nodes have the snapshot laws, and Independent is the product."""
    B, D, m0, S0 = random_ou(3, np.random.default_rng(0))
    laws = [tuple(torch.tensor(a) for a in ou_moments(B, D, m0, S0, t)) for t in [0.0, 0.2, 0.4]]

    means, covs = GaussianOT().fit(laws).moments()
    for k, (mean, cov) in enumerate(laws):
        assert torch.allclose(means[k], mean)
        assert torch.allclose(covs[k, k], cov, atol=1e-10)

    _, covs = Independent().fit(laws).moments()
    assert torch.count_nonzero(covs[0, 1]) == 0

    # the debiased map is the identity between equal laws
    same = [laws[1], laws[1]]
    _, covs = GaussianDebiasedSinkhorn(reg=0.5).fit(same).moments()
    assert torch.allclose(covs[0, 1], laws[1][1], atol=1e-10)


def test_pfm_rounds():
    """PFM_ runs the iterated coupling with path ensembles and records each round."""
    dist = [d.to(torch.float32) for d in _snapshots([200, 200, 200])]
    times = torch.tensor([0.0, 0.3, 0.6])
    X = torch.cat([torch.cat([d, t.expand(d.shape[0], 1)], dim=1) for d, t in zip(dist, times)])
    net = AutonomousODEFlow()
    net.init_model(X)
    history = []
    _, _, loss = PFM_(
        dist,
        times,
        net,
        interp=ChebyshevInterpolant(reg=1e-3),
        paths=DebiasedSinkhorn(reg=1.0),
        bs=64,
        n_epochs=20,
        n_rounds=3,
        n_epochs_round=10,
        history=history,
        verbose=False,
        generator=torch.Generator().manual_seed(0),
    )
    assert len(loss) == 20 + 2 * 10
    assert [h["round"] for h in history] == [1, 2, 3]


def _ou_model(B, D, m0, S0):
    ndim = B.shape[0]
    score = OUScore(m0=torch.tensor(m0), S0=torch.tensor(S0), D=torch.tensor(D))
    score.Ndim_ = ndim
    score.net_ = ConstantNN([ndim, ndim]).to(dtype=torch.float64)
    with torch.no_grad():
        score.net_.B.copy_(torch.tensor(B))
    flow = OUFlow(score=score, D=torch.tensor(D))
    flow.init_model(torch.zeros((1, ndim + 1), dtype=torch.float64))
    return flow.to(dtype=torch.float64)


def test_pfm_linear_recovers_ou():
    """The large-n limit of closed-form PFM with the iterated Gaussian OT coupling recovers B,
    and the finite-n solver gives a close estimate from samples."""
    B, D, m0, S0 = random_ou(3, np.random.default_rng(1))
    times = 0.05 * np.arange(11)

    # n = infinity: exact laws, nothing fitted on data
    B_lim = ou_pfm_limit(
        B,
        D,
        m0,
        S0,
        times,
        GaussianOT(),
        ChebyshevInterpolant(reg=0.0, degree=6, penalty="time-curvature"),
        n_rounds=60,
        anderson=5,
        tol=1e-9,
    )
    assert np.linalg.norm(B_lim - B) / np.linalg.norm(B) < 0.05

    # finite n: samples, the pfm-linear solver
    rng = np.random.default_rng(2)
    samples, _ = sample_ornstein_uhlenbeck(B, D, m0, S0, times, 2000, rng)
    dist = [torch.tensor(s) for s in samples]
    net = _ou_model(B, D, m0, S0)
    PFM_LINEAR_(
        dist,
        torch.tensor(times),
        net,
        interp=LinearInterpolant(),
        paths=GaussianOT(),
        n_rounds=20,
        anderson=5,
        tol=1e-8,
        verbose=False,
    )
    B_hat = net.net_.B.detach().numpy()
    assert np.linalg.norm(B_hat - B) / np.linalg.norm(B) < 0.3
