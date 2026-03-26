import numpy as np
import torch
import torch.nn as nn

from pfi import PFI
from pfi.flow import FlowModel
from pfi.flow.interpolants import ChebyshevInterpolant
from pfi.flow.models import CLEFlow
from pfi.score import ScoreModel
from pfi.utils.data import X_from_snapshots
from pfi.utils.nns import DNN
from pfi.utils.simulations import g_rate, simulate_toggle_switch, toggle_switch


def test_pfi_toggle_switch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model_params = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 4.0])
    ndim = 2
    vol = 4.0
    nsamples = 4000
    K = 5
    Dt = 0.3
    gr = 2.0


    np.random.seed(0)
    torch.manual_seed(0)

    samples_full, tt = simulate_toggle_switch(
        model_params=model_params,
        vol=vol,
        gr=gr,
        nsamples=nsamples,
        ndim=ndim,
        Dt=Dt,
        K=K,
    )
    X = X_from_snapshots(samples_full, tt)

    Np = 50
    Np_flow = 25
    score_model = DNN([ndim + 2, Np, Np, Np, Np, Np, ndim], activation=nn.ELU()).to(device)
    drift_model = DNN([ndim, Np_flow, Np_flow, Np_flow, Np_flow, ndim], activation=nn.ELU(), seed=0).to(device)
    growth_model = DNN([ndim, Np_flow, Np_flow, Np_flow, Np_flow, 1], activation=nn.ELU(), seed=0).to(device)

    score_estimator = ScoreModel(
        model=score_model,
        solver="dsm",
        solver_kwargs=dict(L=10, 
                           n_epochs=2000, 
                           bs=None, 
                           adp_flag=1, 
                           lr=5e-4, 
                           verbose=False),
        device=device,
    )
    flow_estimator = FlowModel(
        flow=CLEFlow(net=drift_model, score=None, Ndim=ndim, vol=vol, lx=1.0),
        growth=growth_model,
        solver="pfm_paper",
        solver_kwargs=dict(
            interp=ChebyshevInterpolant(device=device),
            n_epochs=2000,
            lr=1e-3,
            nb=1,
            fac=2,
            verbose=False,
        ),
        device=device,
    )

    est = PFI(
        score_estimator=score_estimator,
        flow_estimator=flow_estimator,
        fit_on_score_samples=True,
    )
    est.fit(X)

    score_eds = est.score_estimator.score(X)
    assert np.all(score_eds < 0.005)

    Ngrid = 50
    lim = 4.0
    gx = np.linspace(0.0, lim, Ngrid)
    gy = np.linspace(0.0, lim, Ngrid)
    xx, yy = np.meshgrid(gx, gy)
    grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)

    force_true = toggle_switch(grid_xy, model_params)
    grid_xy_t = torch.tensor(grid_xy, dtype=torch.float32, device=device)
    with torch.no_grad():
        force_inf = est.flow_estimator.flow_.net(grid_xy_t).cpu().numpy()

    growth_true = g_rate(grid_xy[:, 0], grid_xy[:, 1], gr)
    with torch.no_grad():
        growth_inf = est.flow_estimator.growth_(grid_xy_t).squeeze(-1).cpu().numpy()

    force_rel_mse = np.mean(((force_true - force_inf) ** 2).sum(axis=1)) / np.mean((force_true ** 2).sum(axis=1))
    growth_rel_mse = np.mean((growth_true - growth_inf) ** 2) / np.mean(growth_true ** 2)

    assert force_rel_mse < 0.07
    assert growth_rel_mse < 0.07
