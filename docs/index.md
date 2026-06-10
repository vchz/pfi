# PFI Documentation

PFI (Probability Flow Inference) provides estimators to learn a score model and a probability-flow model from time-resolved cross-sectional data. The package is organized around a high-level composite estimator API, and a lower-level API for users who want to control the score stage, the flow stage, or the model classes directly.

## Installation

```bash
git clone git@github.com:vchz/pfi.git
cd pfi
pip install -e .
```

Main dependencies include `numpy`, `torch`, `tqdm`, `POT`, `geomloss`, `torchcubicspline`, and `scanpy`.

## Quick Start (composite estimator)

### Data format and loading

All estimators expect a matrix `X` with shape `(n_samples_total, ndim + 1)`:

- columns `0..ndim-1`: state variables
- last column: time

If you already have snapshots grouped by time, convert them with [`X_from_snapshots`](api/utils/data/X_from_snapshots.md):

```python
from pfi.utils.data import X_from_snapshots

X = X_from_snapshots(snaps, times)
```

[`load_data`](api/utils/data/load_data.md) accepts either:

- a local `.h5ad` / `.h5ad.gz` path
- the aliases `"natcomm"` or `"kaggle"`

For these aliases, the data is fetched from the Zenodo repository associated with the package: <https://doi.org/10.5281/zenodo.19237707>. Downloaded files are cached in `pfi.utils.data.PFI_DATA_FOLDER`, which defaults to `~/pfi_data`.

For the exact function signature and outputs, see [`load_data`](api/utils/data/load_data.md).

### Using a composite estimator

The main high-level entry point is [`make_pfi_estimator`](api/make_pfi_estimator.md). It creates a composite [`PFI`](api/PFI.md) object from a score estimator and a flow estimator. The fitting order is fixed: score first, then flow.

```python
import torch
from pfi import make_pfi_estimator
from pfi.flow.models import CLEFlow
from pfi.score.models import DSMScore

device = "cuda" if torch.cuda.is_available() else "cpu"

score_model = DSMScore(
    sizes=[128, 128, 128, 128],
    noise_lvl=0.01,
    batch_norm="zscore",
)

flow_model = CLEFlow(
    score=None,
    sizes=[128, 128, 128],
    lx=1.5,
)

est = make_pfi_estimator(
    score_model=score_model,
    flow_model=flow_model,
    s_solver="dsm",
    f_solver="pfm",
    s_solver_kwargs={"L": 5, "lr": 5e-4, "n_epochs": 4000},
    f_solver_kwargs={"fac": 2, "nb": 1, "lr": 1e-3, "n_epochs": 2000},
    device=device,
    random_state=0,
)
est.fit(X)
```

This is the standard high-level usage pattern. For the complete signature, see [`make_pfi_estimator`](api/make_pfi_estimator.md). The resulting object is documented in [`PFI`](api/PFI.md).

### Quick note on the solvers

The score stage is exposed through [`ScoreEstimator`](api/score/ScoreEstimator.md). The solver currently supported there is:

- `dsm`: denoising score matching, taken from: <https://doi.org/10.1162/NECO_a_00142>, <https://doi.org/10.48550/arXiv.1907.05600>.

The flow stage is exposed through [`FlowEstimator`](api/flow/FlowEstimator.md). The main solver names are:

- `pfm`: the flow-matching solver, please cite: <https://doi.org/10.48550/arXiv.2604.25062>.
- `upfi`: the neural-ODE-style UPFI solver, please cite: <https://doi.org/10.48550/arXiv.2505.13197>, <https://doi.org/10.1073/pnas.2420621122>.

## Low level usage

If you want finer control, work with the two stages separately using [`ScoreEstimator`](api/score/ScoreEstimator.md) and [`FlowEstimator`](api/flow/FlowEstimator.md) directly.

By default `ScoreEstimator` uses [`DSMScore`](api/score/models/DSMScore.md), but you can provide your own instantiated score model. `FlowEstimator` accepts an instantiated flow model and an optional growth model.

```python
from pfi.score import ScoreEstimator
from pfi.flow import FlowEstimator
from pfi.flow.models import CLEFlow, AutonomousGrowth

score_est = ScoreEstimator(
    solver="dsm",
    solver_kwargs={"L": 5, "lr": 5e-4, "n_epochs": 4000},
    device=device,
)
score_est.fit(X)

flow_est = FlowEstimator(
    flow=CLEFlow(score=score_est.model_, lx=0.3),
    growth=AutonomousGrowth(),
    solver="pfm",
    solver_kwargs={"fac": 2, "nb": 1, "lr": 1e-3, "n_epochs": 2000},
    device=device,
)
flow_est.fit(X)
```

The score-stage methods are:

- [`ScoreEstimator.fit`](api/score/ScoreEstimator.md): fit the score model from `(x, t)` data
- [`ScoreEstimator.predict`](api/score/ScoreEstimator.md): evaluate the score on time-augmented inputs
- [`ScoreEstimator.sample`](api/score/ScoreEstimator.md): generate samples at the times present in the input
- [`ScoreEstimator.score`](api/score/ScoreEstimator.md): compare generated samples to observed samples, time by time

The flow-stage methods are:

- [`FlowEstimator.fit`](api/flow/FlowEstimator.md): fit the flow model, and optionally a growth model
- [`FlowEstimator.predict`](api/flow/FlowEstimator.md): evaluate the learned drift
- [`FlowEstimator.sample`](api/flow/FlowEstimator.md): simulate trajectories from initial states
- [`FlowEstimator.score`](api/flow/FlowEstimator.md): push source samples to the next later target time and compare them to target samples

For concrete runnable workflows, use the notebooks in the examples folder:

- <https://github.com/vchz/pfi/tree/main/examples>

For the full public API reference, browse:

- [API reference](api/SUMMARY.md)
