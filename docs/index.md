# PFI Documentation

PFI provides modular tools for score matching, flow regression, and end-to-end pipeline training on snapshot data.

## Installation

```bash
git clone git@github.com:vchz/pfi.git
cd pfi
pip install -e .
```

Main dependencies: `numpy`, `torch`, `tqdm`, `POT`, `geomloss`, `torchcubicspline`.

## Quick Start

### Prepare the data matrix `X`

`X` must have shape `(n_samples_total, ndim + 1)`, with time in the last column.

```python
from pfi.utils.data import X_from_snapshots

# snaps: list of arrays, snaps[k].shape = (n_k, ndim)
# times: array of shape (n_snaps,)
X = X_from_snapshots(snaps, times)
```

### Train a score estimator

```python
import torch
import torch.nn as nn
from pfi.score import ScoreModel
from pfi.utils.nns import DNN

ndim = X.shape[1] - 1
device = "cuda" if torch.cuda.is_available() else "cpu"

score_model = DNN([ndim + 2, 64, 64, 64, ndim], activation=nn.ELU()).to(device)

score_reg = ScoreModel(
    model=score_model,
    solver="dsm",
    solver_kwargs={"L": 10, "n_epochs": 2000, "lr": 1e-3, "bs": None, "adp_flag": 1},
    device=device,
)
score_reg.fit(X)

ed_per_time = score_reg.score(X)
print(ed_per_time)
```

### Train a flow estimator

```python
import torch.nn as nn
from pfi.flow import FlowModel
from pfi.flow.models import CLEFlow
from pfi.flow.interpolants import ChebyshevInterpolant
from pfi.utils.nns import DNN

ndim = X.shape[1] - 1

flow_net = DNN([ndim, 64, 64, 64, ndim], activation=nn.ELU()).to(device)
flow_model = CLEFlow(net=flow_net, score=score_reg.model_, Ndim=ndim, lx=1.0)

flow_reg = FlowModel(
    flow=flow_model,
    growth=None,
    solver="pfm",
    solver_kwargs={"interp": ChebyshevInterpolant(device=device), "n_epochs": 2000, "lr": 1e-3, "fac": 2, "nb": 1},
    device=device,
)
flow_reg.fit(X)
```

## Train the Full PFI Pipeline

You can create a full composite estimator directly with `make_pfi_estimator`, then fit it end-to-end.

```python
from pfi import make_pfi_estimator

ndim = X.shape[1] - 1
pfi_est = make_pfi_estimator(ndim=ndim, device=device, seed=0)
pfi_est.fit(X)
```

To customize, pass a parameter dictionary:

```python
pfi_est = make_pfi_estimator(ndim=ndim, trial_params=params, device=device, seed=0)
```

If you need separated components, use `by_parts=True`:

```python
score_estimator, flow_estimator = make_pfi_estimator(ndim=ndim, trial_params=params, by_parts=True)
```

See API details for parameter definitions:

- [`make_pfi_estimator` API reference](api/pfi/#pfi.make_pfi_estimator)

## Notes

- Current flow solvers exposed by `FlowModel` include `pfm`.
- `upfi` which implements the simulation-based UPFI approach will be implemented soon.
