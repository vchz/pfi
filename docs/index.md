# PFI Documentation

PFI provides modular tools for score estimation, flow estimation, and end-to-end training of a composite estimator.

## Installation

```bash
git clone git@github.com:vchz/pfi.git
cd pfi
pip install -e .
```

Main dependencies include `numpy`, `torch`, `tqdm`, `POT`, `geomloss`, `torchcubicspline`, `scanpy`.

## Quick Start (composite estimator)

### Data format and loading

All estimators expect a matrix `X` of shape `(n_samples_total, ndim + 1)`:

- columns `0..ndim-1`: state/features
- last column: time

You can build this format from snapshots with:

```python
from pfi.utils.data import X_from_snapshots

# snaps[k]: array of shape (n_k, ndim)
# times: array of snapshot times in the same order
X = X_from_snapshots(snaps, times)
```

`pfi.utils.data.load_data` accepts:

- a local path compatible with Scanpy (`.h5ad` or `.h5ad.gz`)
- aliases `"natcomm"` or `"kaggle"`

For aliases, data is fetched from the package Zenodo repository:
[https://doi.org/10.5281/zenodo.19237707](https://doi.org/10.5281/zenodo.19237707) and downloaded files are cached in `pfi.utils.data.PFI_DATA_FOLDER` (default `~/pfi_data`).

### Using a composite estimator

Use `make_pfi_estimator` for the standard pipeline (fit score, then fit flow):

```python
import torch
from pfi import make_pfi_estimator

device = "cuda" if torch.cuda.is_available() else "cpu"
ndim = X.shape[1] - 1

pfi_est = make_pfi_estimator(
    ndim=ndim,
    params=None,   # uses pfi.DEFAULT_PFI_PARAMETERS
    device=device,
    seed=0,
)
pfi_est.fit(X)
```

You can customize networks, flow model, solvers, and solver kwargs via `params`. The default params are in `pfi.DEFAULT_PFI_PARAMETERS`. Typical example:

```python
import torch.nn as nn
from pfi.utils.nns import DNN, SpectralNormDNN
from pfi.flow.models import CLEFlow
from pfi.flow.interpolants import LinearInterpolant

params = {
    "s_solver": "dsm",
    "f_solver": "pfm",
    "f_model": CLEFlow,
    "f_model_kwargs": {"lx": 0.3},
    "s_net": SpectralNormDNN,
    "f_net": SpectralNormDNN,
    "g_net": None,
    "s_net_kwargs": {"activation": nn.ELU(), "feature_norm": False},
    "f_net_kwargs": {"activation": nn.ELU(), "feature_norm": True},
    "s_width": 128,
    "s_depth": 4,
    "f_width": 128,
    "f_depth": 3,
    "s_noise_lvl": 0.01,
    "s_solver_kwargs": {"L": 5, "adp_flag": 0},
    "f_solver_kwargs": {"fac": 4, "nb": 1, "interp": LinearInterpolant(), "bs": 512},
    "s_lr": 5e-4,
    "f_lr": 1e-3,
    "s_n_epochs": 4000,
    "f_n_epochs": 1500,
    "fit_on_score_samples": False,
}
```
If you specify any parameter when building the estimator, it will update this dictionnary, except for any nested dictionnary which is replaced with the user chosen values.

### Quick note on the solvers

There is only one score solver for now: `dsm`, which impleements denoising score matching, as introduced in [https://doi.org/10.48550/arXiv.1907.05600](https://doi.org/10.48550/arXiv.1907.05600) and [https://doi.org/10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142).


Flow solvers:

- `upfi`: UPFI/PFI-style formulation (see [https://doi.org/10.48550/arXiv.2505.13197](https://doi.org/10.48550/arXiv.2505.13197) and [https://doi.org/10.1073/pnas.2420621122](https://doi.org/10.1073/pnas.2420621122))
- `pfm`: unbalanced flow matching used in this package (publication in preparation)
- `external.*`: wrappers for external methods used for benchmarking (currently `external.deepruotv2`)
- `future.*`: experimental approaches not fully tested (currently `future.ufm_uot`, `future.ufm_ot`)

### Hyperparameter optimization

`hyperopt_pfi(X, n_trials, search_space, ...)` runs Optuna-based multi-objective tuning for score and flow objectives.  
It is available and usable, but not yet fully validated across all solver/model combinations.

## Low level usage

If you do not use `make_pfi_estimator` because you want to check first the quality of the score for instance, the expected order is:

- fit a `ScoreModel`
- freeze the fitted score at a given noise level when using `dsm`
- instantiate and fit a `FlowModel` that uses that frozen score

Example:

```python
from pfi.score import ScoreModel, freeze_dsm_score
from pfi.flow import FlowModel
from pfi.flow.models import CLEFlow

# 1) Fit score
score_reg = ScoreModel(
    model=score_net,
    solver="dsm",
    solver_kwargs={"L": 5, "n_epochs": 4000, "lr": 5e-4},
    noise_lvl=0.01,
    device=device,
)
score_reg.fit(X)

# 2) Freeze score (DSM)
frozen_score = freeze_dsm_score(score_reg.model_, noise_lvl=score_reg.noise_lvl_)

# 3) Build + fit flow
flow_model = CLEFlow(net=flow_net, score=frozen_score, Ndim=ndim, lx=0.3)
flow_reg = FlowModel(
    flow=flow_model,
    growth=growth_model,  # or None
    solver="pfm",
    solver_kwargs={"n_epochs": 1500, "lr": 1e-3, "fac": 4, "nb": 1},
    device=device,
)
flow_reg.fit(X)
```

Importantly, this package is modular which means that you can try a variety of different models, gradient, CLE, autonomous, additive noise. All these models are detailed in [API Reference](api/pfi/flow/models.md)

In the spirit of sklearn, the FlowModel and ScoreModel implement different other methods:

- `ScoreModel.sample(X)` generates samples at the snapshot times of `X`.
- `ScoreModel.score(X)` returns per-time energy distances between generated and observed samples.
- `FlowModel.sample(X0, Dt, dt, stoch, pos)` simulates trajectories from initial states `X0`. If `stoch` is `True` it simulates stochastic trajectories with Euler-Maruyama. Otherwise it simulates the probability flow trajectories.
- `FlowModel.score(X, Y, ...)` pushes each source time in `X` to the next strictly later time in `Y` and computes energy-distance errors.

For full API details and runnable notebooks, use:

- API docs under [API Reference](api/summary.md)
- examples under [https://github.com/vchz/pfi/tree/main/examples](https://github.com/vchz/pfi/tree/main/examples)
