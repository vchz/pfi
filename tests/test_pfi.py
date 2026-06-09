import numpy as np
import torch
import torch.nn as nn

from pfi import make_pfi_estimator
from pfi.flow.interpolants import ChebyshevInterpolant
from pfi.flow.models import CLEFlow, AutonomousGrowth
from pfi.score.models import DSMScore
from pfi.utils.data import X_from_snapshots
from pfi.utils.simulations import g_rate, simulate_toggle_switch, toggle_switch

TOGGLE_SWITCH_MODEL_PARAMS = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 4.0])
TOGGLE_SWITCH_GROWTH = 2.0
TOGGLE_SWITCH_VOL = 4.0

def _get_toggle_switch_data(seed):
    """Generate toggle-switch training data for a fixed simulation setup.

    Parameters
    ----------
    seed : int
        Random seed used for the simulation.

    Returns
    -------
    ndarray
        Time-augmented toggle-switch samples.
    """
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
    """Compute relative MSE on force and growth over a fixed grid.

    Parameters
    ----------
    flow_estimator : pfi.flow.FlowEstimator
        Fitted flow estimator.

    Returns
    -------
    tuple of float
        Relative MSE on the inferred force and growth.
    """
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
        force_inf = flow_estimator.flow_.evaluate_net(grid_xy_t).cpu().numpy()

    growth_true = g_rate(grid_xy[:, 0], grid_xy[:, 1], TOGGLE_SWITCH_GROWTH)
    with torch.no_grad():
        growth_inf = flow_estimator.growth_.evaluate_net(grid_xy_t).squeeze(-1).cpu().numpy()

    force_rel_mse = np.mean(((force_true - force_inf) ** 2).sum(axis=1)) / np.mean((force_true ** 2).sum(axis=1))
    growth_rel_mse = np.mean((growth_true - growth_inf) ** 2) / np.mean(growth_true ** 2)
    return force_rel_mse, growth_rel_mse

def test_pfm_toggle_switch():
    """Fit the composite estimator with PFM on toggle-switch data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = _get_toggle_switch_data(seed=0)

    Np = 50
    Np_flow = 25
    score_model = DSMScore(
        sizes=[Np] * 5,
        spectral_norm=False,
        activation=nn.ELU(),
        feature_norm=None,
        batch_norm="zscore",
        noise_lvl=0.1,
    )
    flow_model = CLEFlow(
        score=None,
        sizes=[Np_flow] * 4,
        spectral_norm=False,
        activation=nn.ELU(),
        lx=1.0,
        feature_norm=None,
        batch_norm=None,
    )
    growth_model = AutonomousGrowth(
        sizes=[Np_flow] * 4,
        spectral_norm=False,
        activation=nn.ELU(),
        feature_norm=None,
        batch_norm=None,
    )

    est = make_pfi_estimator(
        score_model=score_model,
        flow_model=flow_model,
        growth_model=growth_model,
        s_solver="dsm",
        f_solver="pfm",
        s_solver_kwargs={"L": 10,
                         "bs": None,
                         "adp_flag": 1,
                         "lr": 5e-4,
                         "n_epochs": 2000},
        f_solver_kwargs={
            "interp": ChebyshevInterpolant(),
            "nb": 1,
            "fac": 2,
            "lr": 1e-3,
            "n_epochs": 2000,
        },
        device=device,
        random_state=0,
    )
    est.flow_estimator.flow.vol = TOGGLE_SWITCH_VOL
    est.fit(X)

    score_eds = est.score_estimator_.score(X)
    assert np.all(score_eds < 0.005)

    force_rel_mse, growth_rel_mse = _get_toggle_switch_errors(est.flow_estimator_)
    assert force_rel_mse < 0.08
    assert growth_rel_mse < 0.05

def test_upfi_toggle_switch():
    """Fit the composite estimator with UPFI on toggle-switch data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = _get_toggle_switch_data(seed=0)

    Np = 50
    Np_flow = 25
    score_model = DSMScore(
        sizes=[Np] * 5,
        spectral_norm=False,
        activation=nn.ELU(),
        feature_norm=None,
        batch_norm="zscore",
        noise_lvl=0.1,
    )
    flow_model = CLEFlow(
        score=None,
        sizes=[Np_flow] * 4,
        spectral_norm=False,
        activation=nn.ELU(),
        lx=1.0,
        feature_norm=None,
        batch_norm=None,
    )
    growth_model = AutonomousGrowth(
        sizes=[Np_flow] * 4,
        spectral_norm=False,
        activation=nn.ELU(),
        feature_norm=None,
        batch_norm=None,
    )

    est = make_pfi_estimator(
        score_model=score_model,
        flow_model=flow_model,
        growth_model=growth_model,
        s_solver="dsm",
        f_solver="upfi",
        s_solver_kwargs={"L": 10,
                         "bs": None,
                         "adp_flag": 1,
                         "lr": 5e-4,
                         "n_epochs": 2000},
        f_solver_kwargs={"n_steps": 2,
                         "reg_wfr": 1,
                         "reach": 20.0,
                         "lr": 1e-3,
                         "n_epochs": 2000},
        device=device,
        random_state=0,
    )
    est.flow_estimator.flow.vol = TOGGLE_SWITCH_VOL
    est.fit(X)

    force_rel_mse, growth_rel_mse = _get_toggle_switch_errors(est.flow_estimator_)
    assert force_rel_mse < 0.07
    assert growth_rel_mse < 0.09
