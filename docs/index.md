# PFI Documentation

PFI provides modular tools for:
- score estimation (`pfi.score.ScoreModel`)
- flow estimation (`pfi.flow.FlowModel`)
- end-to-end composition (`pfi.PFI`, `make_pfi_estimator`)

## Installation

```bash
git clone git@github.com:vchz/pfi.git
cd pfi
pip install -e .
```

Main dependencies include `numpy`, `torch`, `tqdm`, `POT`, `geomloss`, `torchcubicspline`, `scanpy`.

## Data Format

All estimators expect a matrix `X` of shape `(n_samples_total, ndim + 1)`:
- columns `0..ndim-1`: state/features
- last column: time

You can build this format from a list of snapshots (NumPy arrays), one per
time point:

```python
from pfi.utils.data import X_from_snapshots

# snaps[k]: array of shape (n_k, ndim)
# times: array of snapshot times, same order as snaps
X = X_from_snapshots(snaps, times)
```

## Load Datasets

`pfi.utils.data.load_data` accepts either:
- a local path compatible with scanpy 
- the aliases `"natcomm"` or `"kaggle"`

For aliases (`"natcomm"`/`"kaggle"`), data is fetched from the Zenodo
repository associated with this package:
https://doi.org/10.5281/zenodo.19237707

Downloaded files are cached in `pfi.utils.data.PFI_DATA_FOLDER` (default `~/pfi_data`).

## Quick Start: Composite Estimator

The recommended entry point is the composite estimator returned by
`make_pfi_estimator`. It handles:
- score fitting first
- flow fitting second

```python
import torch
from pfi import make_pfi_estimator

device = "cuda" if torch.cuda.is_available() else "cpu"
ndim = X.shape[1] - 1

pfi_est = make_pfi_estimator(
    ndim=ndim,
    params=None,   # uses DEFAULT_PFI_PARAMETERS
    device=device,
    seed=0,
)
pfi_est.fit(X)
```

## Composite Estimator Customization

You can customize networks, flow model, solvers, optimization hyperparameters,
and solver-specific settings through `params`.

Example:

```python
import torch.nn as nn
from pfi import make_pfi_estimator
from pfi.utils.nns import DNN, SpectralNormDNN
from pfi.flow.models import PositiveCLEFlow
from pfi.flow.interpolants import ChebyshevInterpolant

params = {
    "s_solver": "dsm",
    "f_solver": "upfi",
    "f_model": PositiveCLEFlow,
    "f_model_kwargs": {"lx": 0.5},
    "s_net": SpectralNormDNN,
    "f_net": SpectralNormDNN,
    "g_net": DNN,
    "s_net_kwargs": {"activation": nn.ELU(), "feature_norm": False},
    "f_net_kwargs": {"activation": nn.ELU(), "feature_norm": True},
    "s_width": 128,
    "s_depth": 4,
    "f_width": 128,
    "f_depth": 3,
    "s_noise_lvl": 0.01,
    "s_solver_kwargs": {"L": 5, "adp_flag": 0},
    "f_solver_kwargs": {"n_steps": 3, "reg_wfr": 0.1, "reach": 20.0},
    "s_lr": 5e-4,
    "f_lr": 1e-3,
    "s_n_epochs": 4000,
    "f_n_epochs": 1500,
    "fit_on_score_samples": False,
}
pfi_est = make_pfi_estimator(ndim=ndim, params=params, device=device, seed=0)
pfi_est.fit(X)
```

## Factory Functions

Top-level helpers exported by `pfi`:
- `make_score_estimator(ndim, params=None, device="cpu", seed=0, verbose=True)`
- `make_flow_estimator(ndim, params=None, device="cpu", seed=0, verbose=True, score=None)`
- `make_pfi_estimator(ndim, params=None, device="cpu", seed=0, verbose=True)`
- `hyperopt_pfi(X, n_trials=50, search_space=None, device="cpu", seed=0)`

Default parameter dictionary:
- `pfi.DEFAULT_PFI_PARAMETERS`

### `params` dictionary: expected keys

The factory functions read a single `params` dictionary (missing keys fall back
to `DEFAULT_PFI_PARAMETERS`, except for the `s_solver_kwargs` and `f_solver_kwargs` which are overwritten as soon as they are set by the user).

Main keys:
- `s_solver`, `f_solver`
- `s_net`, `f_net`, `g_net` (network classes; `g_net=None` disables growth net)
- `s_net_kwargs`, `f_net_kwargs` (kwargs passed to network constructors)
- `s_width`, `s_depth`, `f_width`, `f_depth`
- `s_noise_lvl`
- `s_solver_kwargs`, `f_solver_kwargs`
- `f_model` (flow model class)
- `f_model_kwargs` (e.g. `{"lx": ...}`)
- `s_lr`, `f_lr`, `s_n_epochs`, `f_n_epochs`
- `fit_on_score_samples`

Typical example:

```python
params = {
    "s_solver": "dsm",
    "f_solver": "pfm",
    "f_model": CLEFlow,
    "f_model_kwargs": {"lx": 0.3}, # sets the decay rate at 0.3 1/day
    "s_net": SpectralNormDNN,
    "f_net": SpectralNormDNN,
    "g_net": None,
    "s_net_kwargs": {"activation": nn.ELU(), "feature_norm": False},
    "f_net_kwargs": {"activation": nn.ELU(), "feature_norm": True},
    "s_solver_kwargs": {"L": 5, "adp_flag": 0},
    "f_solver_kwargs": {"fac": 4, "nb": 1, "interp": LinearInterpolant(), "bs": 512},
    "s_width": 128,
    "s_depth": 4,
    "f_width": 128,
    "f_depth": 3,
    "s_noise_lvl": 0.01,
    "s_lr": 5e-4,
    "f_lr": 1e-3,
    "s_n_epochs": 4000,
    "f_n_epochs": 1500,
    "fit_on_score_samples": False,
}
```

## Solver Names

Current `FlowModel` solver keys:
- `"pfm"`
- `"upfi"`
- `"future.ufm_uot"`
- `"future.ufm_ot"`
- `"external.deepruotv2"`

Current `ScoreModel` solver key:
- `"dsm"`

## Solver Families

- `upfi`: implements the PFI and UPFI algorithms presented in:
  - https://doi.org/10.48550/arXiv.2505.13197
  - https://doi.org/10.1073/pnas.2420621122
- `pfm`: implements the unbalanced flow-matching solver used in this package
  (publication in preparation). This is the fastest and more stable solver.
- `external.*`: wrappers around external solvers for benchmarking.  
  Currently available: `external.deepruotv2`. More wrappers will be added.
- `future.*`: experimental approaches that are not yet fully tested.

## Hyperparameter Optimization

`hyperopt_pfi(X, n_trials, search_space, ...)` is available, but not yet fully
validated across all solver/model configurations.

## Examples

For runnable end-to-end notebooks, see:
- https://github.com/vchz/pfi/tree/main/examples

## Low-Level Usage

If you do not use `make_pfi_estimator`, the expected sequence is:
1. fit a `ScoreModel`
2. freeze the fitted score (for `dsm`)
3. build and fit a `FlowModel` using that frozen score

### 1) Fit score model

```python
from pfi.score import ScoreModel

score_reg = ScoreModel(
    model=score_net,
    solver="dsm",  # currently the only score solver
    solver_kwargs={"L": 5, "n_epochs": 4000, "lr": 5e-4},
    noise_lvl=0.01,
    device=device,
)
score_reg.fit(X)
```

`ScoreModel.sample(X)` generates samples at the same snapshot times as `X`
using the fitted score model.  
`ScoreModel.score(X)` returns per-time energy-distance values between generated
and observed samples.

### 2) Freeze score (DSM)

For `dsm`, the flow models should receive a frozen score callable:

```python
from pfi.score import freeze_dsm_score

frozen_score = freeze_dsm_score(score_reg.model_, noise_lvl=score_reg.noise_lvl_)
```

### 3) Build and fit flow model

```python
from pfi.flow import FlowModel
from pfi.flow.models import CLEFlow

flow_model = CLEFlow(
    net=flow_net,
    score=frozen_score,
    Ndim=ndim,
    lx=0.3,
)

flow_reg = FlowModel(
    flow=flow_model,
    growth=growth_model,  # or None
    solver="pfm",
    solver_kwargs={"n_epochs": 1500, "lr": 1e-3, "fac": 4, "nb": 1},
    device=device,
)
flow_reg.fit(X)
```

`FlowModel.sample(X0, Dt, dt, stoch, pos)` simulates trajectories from initial
states `X0` for horizon `Dt`.  
`FlowModel.score(X, Y, ...)` computes per-time energy-distance values by pushing
`X` to target times and comparing against `Y`.

## API Reference

- [`pfi` API](api/pfi/summary.md)
