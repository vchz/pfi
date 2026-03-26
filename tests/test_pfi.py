import numpy as np
import torch
import torch.nn as nn

from pfi import make_pfi_estimator
from pfi.flow.interpolants import ChebyshevInterpolant
from pfi.flow.models import CLEFlow
from pfi.utils.data import X_from_snapshots
from pfi.utils.nns import DNN
from pfi.utils.simulations import g_rate, simulate_toggle_switch, toggle_switch

TOGGLE_SWITCH_MODEL_PARAMS = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 4.0])
TOGGLE_SWITCH_GROWTH = 2.0
TOGGLE_SWITCH_VOL = 4.0


def _get_toggle_switch_data(seed):
    """Generate toggle-switch training data for a fixed simulation setup."""
    ndim = 2
    nsamples = 4000
    k_snaps = 5
    dt_snap = 0.3

    np.random.seed(seed)
    torch.manual_seed(seed)
    samples_full, tt = simulate_toggle_switch(
        model_params=TOGGLE_SWITCH_MODEL_PARAMS,
        vol=TOGGLE_SWITCH_VOL,
        gr=TOGGLE_SWITCH_GROWTH,
        nsamples=nsamples,
        ndim=ndim,
        Dt=dt_snap,
        K=k_snaps,
    )
    return X_from_snapshots(samples_full, tt)


def _get_toggle_switch_errors(flow_estimator):
    """Compute relative MSE on force and growth over a fixed grid."""
    n_grid = 50
    lim = 4.0
    gx = np.linspace(0.0, lim, n_grid)
    gy = np.linspace(0.0, lim, n_grid)
    xx, yy = np.meshgrid(gx, gy)
    grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)

    force_true = toggle_switch(grid_xy, TOGGLE_SWITCH_MODEL_PARAMS)
    ref_param = next(flow_estimator.flow_.parameters())
    grid_xy_t = torch.tensor(grid_xy, dtype=ref_param.dtype, device=ref_param.device)
    with torch.no_grad():
        force_inf = flow_estimator.flow_.net(grid_xy_t).cpu().numpy()

    growth_true = g_rate(grid_xy[:, 0], grid_xy[:, 1], TOGGLE_SWITCH_GROWTH)
    with torch.no_grad():
        growth_inf = flow_estimator.growth_(grid_xy_t).squeeze(-1).cpu().numpy()

    force_rel_mse = np.mean(((force_true - force_inf) ** 2).sum(axis=1)) / np.mean((force_true ** 2).sum(axis=1))
    growth_rel_mse = np.mean((growth_true - growth_inf) ** 2) / np.mean(growth_true ** 2)
    return force_rel_mse, growth_rel_mse


def test_pfm_toggle_switch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ndim = 2
    X = _get_toggle_switch_data(seed=0)

    Np = 50
    Np_flow = 25
    params = {
        "s_solver": "dsm",
        "f_solver": "pfm",
        "f_model": CLEFlow,
        "f_model_kwargs": {
            "lx": 1.0,
        },
        "s_net_kwargs": {"feature_norm": False, 
                         "activation": nn.ELU(),
        },
        "f_net_kwargs": {"feature_norm": False, 
                         "activation": nn.ELU(),
        },
        "s_net": DNN,
        "f_net": DNN,
        "g_net": DNN,
        "s_width": Np,
        "f_width": Np_flow,
        "s_depth": 5,
        "f_depth": 4,
        "s_noise_lvl": 0.1,
        "s_solver_kwargs": {
            "L": 10,
            "bs": None,
            "adp_flag": 1,
        },
        "f_solver_kwargs": {
            "interp": ChebyshevInterpolant(),
            "nb": 1,
            "fac": 2,
        },
        "s_lr": 5e-4,
        "f_lr": 1e-3,
        "s_n_epochs": 2000,
        "f_n_epochs": 2000,
        "fit_on_score_samples": False,
    }

    est = make_pfi_estimator(
        ndim=ndim,
        params=params,
        device=device,
        seed=0,
        verbose=False,
    )
    est.flow_estimator.flow.vol = TOGGLE_SWITCH_VOL
    est.fit(X)

    score_eds = est.score_estimator.score(X)
    assert np.all(score_eds < 0.005)

    force_rel_mse, growth_rel_mse = _get_toggle_switch_errors(est.flow_estimator)
    assert force_rel_mse < 0.07
    assert growth_rel_mse < 0.07


def test_upfi_toggle_switch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ndim = 2
    X = _get_toggle_switch_data(seed=0)

    Np = 50
    Np_flow = 25
    params = {
        "s_solver": "dsm",
        "f_solver": "upfi",
        "f_model": CLEFlow,
        "f_model_kwargs": {
            "lx": 1.0,
        },
        "s_net_kwargs": {
            "feature_norm": False, 
            "activation": nn.ELU(),
        },
        "f_net_kwargs": {
            "feature_norm": False, 
            "activation": nn.ELU(),
        },
        "s_net": DNN,
        "f_net": DNN,
        "g_net": DNN,
        "s_width": Np,
        "f_width": Np_flow,
        "s_depth": 5,
        "f_depth": 4,
        "s_noise_lvl": 0.1,
        "s_solver_kwargs": {
            "L": 10,
            "bs": None,
            "adp_flag": 1,
        },
        "f_solver_kwargs": {
            "n_steps": 2,
            "reg_wfr": 0.1,
            "reach": 20.0,
        },
        "s_lr": 5e-4,
        "f_lr": 1e-3,
        "s_n_epochs": 2000,
        "f_n_epochs": 2000,
        "fit_on_score_samples": False,
    }

    est = make_pfi_estimator(
        ndim=ndim,
        params=params,
        device=device,
        seed=0,
        verbose=False,
    )
    est.flow_estimator.flow.vol = TOGGLE_SWITCH_VOL
    est.fit(X)

    force_rel_mse, growth_rel_mse = _get_toggle_switch_errors(est.flow_estimator)
    assert force_rel_mse < 0.06
    assert growth_rel_mse < 0.1
