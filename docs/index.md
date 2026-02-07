# PFI Documentation

PFI provides modular tools for score matching and flow regression on snapshot data.
This documentation explains installation, basic usage, validation, and customization.

## Installation

Clone the repository and install in editable mode:

```bash
git clone <your-repo-url>
cd pfi
pip install -e .
```
The package currently depends on the following other packages: `numpy, torch, tqdm, POT, geomloss, torchcubicspline`.

## How To Use

In the `examples/` folder of the repository we provide two low-dimensional examples to get your hands on the package and how to perform probability flow inference. We summarize briefly below the few key points to work with this code.

### Prepare the data matrix `X`

`X` must be a 2D array of shape `(n_samples_total, ndim + 1)`.
The last column is time, and the first `ndim` columns are state coordinates.

```python
import numpy as np
from pfi.utils import X_from_snapshots

# snaps: list of arrays, snaps[k].shape = (n_k, ndim)
# times: array of shape (n_snaps,)
X = X_from_snapshots(snaps, times)
```

We propose the PFI approach as a mean to fit arbitrary Fokker-Planck Equation (FPE) to such snapshots data. In brief, us and others showed that it amounts to fitting a flow model which depends on the drift of the FPE, but also on the gradient-log probability of the data, also known as score. We illustrate here how to do this for a specific model of Fokker-Planck Equation describing constitutive transcriptional dynamics of gene expression. However, **this package allows to fit any flow model**, even flow models that do not depend on the score.

### Train a score model

```python
import torch
import torch.nn as nn
from pfi.utils import DNN
from pfi.score import ScoreMatching

ndim = X.shape[1] - 1
device = "cuda" if torch.cuda.is_available() else "cpu"

score_model = DNN([ndim + 2, 64, 64, 64, ndim], activation=nn.ELU()).to(device)

score_reg = ScoreMatching(
    model=score_model,
    solver="dsm",
    solver_kwargs={"L": 10, "n_epochs": 2000, "lr": 1e-3, "bs": None, "adp_flag": 1},
    device=device,
)
score_reg.fit(X)
```

## Validate The Score

`ScoreMatching.score(X)` computes per-time energy distance between generated and observed samples.
Lower values indicate better match.

```python
ed_per_time = score_reg.score(X)
print("Energy distance per time:", ed_per_time)
print("Mean energy distance:", ed_per_time.mean())
```
You can also sample using the computed score. For the `solver='dsm'` the sampling is done with an annealed langevin dynamics scheme. A simple langevin dynamics could also do the trick.


### Choose an interpolant

```python
from pfi.flow.interpolants import ChebyshevInterpolant

interp = ChebyshevInterpolant(device=device)
```

Other available interpolants include `LinearInterpolant` and `SplineInterpolant`.

### Regress a flow model

```python
from pfi.flow import FlowRegression
from pfi.flow.models import CLEFlow

drift_model = DNN([ndim, 64, 64, ndim], activation=nn.ELU()).to(device)
flow_model = CLEFlow(
    drift_model=drift_model,
    score=score_reg.model_,
    Ndim=ndim,
    vol=1.0,
    lx=1.0,
)

flow_reg = FlowRegression(
    interp=interp,
    model=flow_model,
    growth_model=None,
    solver="fm",
    solver_kwargs={"n_epochs": 2000, "lr": 1e-3, "fac": 1},
    device=device,
)
flow_reg.fit(X)
```

## Long-term goal of this package

The package is designed for systematic model and solver comparisons. Currently, it is explicitely modular for the choice of the flow model, as long as it is an `nn.Module` which accepts input in of size`(batch_size, ndim+1)` with time as last variable. **Models do not have** to depend on the score, they can very well be any neural network which satisfies the aforementionned requirements.

For the flow regression step, only the flow matching (`solver='fm'`) is available. We will work to implement other solvers (often less scalable) which we and other proposed in previous research papers. For systematic and benchmark and comparison purposes we plan to implement ode, sde and cnf based solvers for comparison purposes.

For the score matching step, only the denoising score matching (`solver='dsm'`) is available. This is the most scalable solver we know to compute the score in high dimensions, but for benchmarking purposes we plan to implement other solvers like denoising score matching.

We will work at updating this package by providing examples on single-cell RNA-seq data, with benchmarks and comparison.

