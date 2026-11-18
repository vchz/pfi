"""Path ensembles: trajectories through the snapshot times, consumed by the flow solvers.

Every method builds a weighted ensemble of paths, one node per snapshot time, from the
snapshots and an optional reference dynamics ``push`` (the current model's flow over one
interval; ``None`` for plain PFM). All classes share the interface of
:class:`PathEnsemble`, so the solvers can use any of them interchangeably::

    paths, masses = ensemble.fit(dist, push=push).sample()

Two families, named after the two objects of optimal transport:

- :class:`PlanPaths` (transport plans): random paths through the observed cells, a Markov
  chain whose transitions are the plans between consecutive (pushed) snapshots. The
  snapshot marginals are reproduced exactly. Subclasses: :class:`Independent`,
  :class:`ExactOT`, :class:`Sinkhorn`, :class:`UnbalancedSinkhorn`. With ``push=None`` the
  plan couplings give the same trajectories as ``couplings.pairwise_OT`` (same random
  draws for one mini-batch, the same construction for several).
- :class:`MapPaths` (transport maps): deterministic paths, a cloud of nodes moved from one
  snapshot time to the next by a map. The nodes are not observed cells and the snapshot
  marginals are only approximately reproduced. Subclasses: :class:`DebiasedSinkhorn`,
  :class:`GaussianOT`, :class:`GaussianDebiasedSinkhorn`.

The Gaussian ensembles (and :class:`Independent`) also accept the exact snapshot laws, a
list of ``(mean, cov)`` pairs, instead of samples: their paths are then Gaussian and
:meth:`PathEnsemble.moments` returns their exact moments (the ``n = infinity`` limit).
"""

from abc import ABC, abstractmethod

import ot
import torch

from .couplings import _compute_pairwise_OT_plan, _sample_trajectory


class PathEnsemble(ABC):
    """Abstract base class of path ensembles.

    Attributes
    ----------
    weighted : bool
        ``True`` if the paths carry masses that encode growth information (required by
        unbalanced solvers). Otherwise the masses returned by :meth:`sample` are the
        snapshot-size ratios (ones for snapshots of equal sizes).
    """

    weighted = False

    @abstractmethod
    def fit(self, dist, push=None, generator=None):
        """Build what the ensemble needs from the snapshots.

        Parameters
        ----------
        dist : list of torch.Tensor, length nsnaps
            ``dist[k]`` has shape ``(n_k, ndim)``, on a common device and dtype.
        push : callable or None, default=None
            Reference dynamics: ``push(x, k)`` returns the positions at ``t_{k+1}`` of
            points ``x`` of shape ``(m, ndim)`` at ``t_k``. ``None`` is the identity
            (plain PFM).
        generator : torch.Generator or None, default=None
            Optional RNG for reproducible sub-sampling.

        Returns
        -------
        self : PathEnsemble
            Fitted ensemble.
        """

    @abstractmethod
    def sample(self, n_paths=None, generator=None):
        """Return paths and their masses.

        Parameters
        ----------
        n_paths : int or None, default=None
            Number of paths. If ``None``, the ensemble's default.
        generator : torch.Generator or None, default=None
            Optional RNG for reproducible sampling.

        Returns
        -------
        paths : torch.Tensor of shape (n_paths, nsnaps, ndim)
            Paths in trajectory-first layout, as expected by the interpolants.
        masses : torch.Tensor of shape (n_paths, nsnaps)
            Mass of every path at every snapshot time.
        """

    def moments(self, n_paths=None, generator=None):
        """First two moments of the path nodes.

        Empirical moments of :meth:`sample` (normalized by the number of paths).
        Ensembles fitted on exact laws override this with the exact moments.

        Parameters
        ----------
        n_paths : int or None, default=None
            Number of paths passed to :meth:`sample`.
        generator : torch.Generator or None, default=None
            Optional RNG for reproducible sampling.

        Returns
        -------
        means : torch.Tensor of shape (nsnaps, ndim)
            Mean of the nodes at every snapshot time.
        covs : torch.Tensor of shape (nsnaps, nsnaps, ndim, ndim)
            ``covs[j, k]`` is the covariance between the nodes at times ``j`` and ``k``.
        """
        paths, _ = self.sample(n_paths, generator=generator)
        means = paths.mean(dim=0)
        centered = paths - means
        covs = torch.einsum("njd,nke->jkde", centered, centered) / paths.shape[0]
        return means, covs

    @staticmethod
    def _is_laws(dist):
        """True if ``dist`` is a list of ``(mean, cov)`` pairs rather than samples."""
        return isinstance(dist[0], (tuple, list))

    @staticmethod
    def _check(dist):
        """Validate the snapshot list and return it."""
        if len(dist) < 2:
            raise ValueError("At least two snapshots are required.")

        ndim = dist[0].shape[1]
        device = dist[0].device
        dtype = dist[0].dtype
        for snap in dist:
            if snap.ndim != 2 or snap.shape[1] != ndim:
                raise ValueError("All snapshots must have shape (n_k, ndim) with a common ndim.")
            if snap.device != device or snap.dtype != dtype:
                raise ValueError("All snapshots must share device and dtype.")
        return dist

    @staticmethod
    def _push(x, k, push):
        """Apply the reference dynamics over interval ``k`` (identity if ``None``)."""
        if push is None:
            return x
        return push(x, k)


# ----------------------------------------------------------------------------------------
# Transport plans: random paths through the observed cells
# ----------------------------------------------------------------------------------------

class PlanPaths(PathEnsemble):
    """Random paths through the observed cells, drawn from plans between snapshots.

    On each of ``n_batches`` mini-batches, ``min_k n_k // n_batches`` cells are
    sub-sampled in every snapshot, and :meth:`plan` couples the (pushed) cells of
    snapshot k with the cells of snapshot k + 1. Paths start uniformly in the first
    snapshot and draw each next cell from the current cell's row of the plan (Markov
    gluing), as in ``couplings.pairwise_OT``, whose helpers are used here.

    Parameters
    ----------
    n_batches : int, default=1
        Number of mini-batches.

    Attributes
    ----------
    batches_ : list of tuple
        For every mini-batch, the sub-sampled cells of each snapshot and the plans.
    batch_size_ : int
        Number of cells per snapshot in a mini-batch.
    masses_ : torch.Tensor of shape (nsnaps,)
        Total mass of each snapshot.
    """

    def __init__(self, n_batches=1):
        self.n_batches = n_batches

    @abstractmethod
    def plan(self, src, tgt, mass_src, mass_tgt):
        """Plan between source and target cells.

        Parameters
        ----------
        src : torch.Tensor of shape (n_src, ndim)
            Source cells (pushed cells of snapshot k).
        tgt : torch.Tensor of shape (n_tgt, ndim)
            Target cells (cells of snapshot k + 1).
        mass_src, mass_tgt : float or torch.Tensor
            Total masses of the source and target snapshots.

        Returns
        -------
        torch.Tensor of shape (n_src, n_tgt)
            Plan normalized to sum to one.
        """

    def _snapshot_masses(self, dist):
        """Total mass of each snapshot (ones when balanced)."""
        return torch.ones(len(dist), dtype=dist[0].dtype, device=dist[0].device)

    def fit(self, dist, push=None, generator=None):
        dist = self._check(dist)
        bs = max(1, min(snap.shape[0] for snap in dist) // self.n_batches)
        masses = self._snapshot_masses(dist)

        self.batches_ = []
        for _ in range(self.n_batches):
            chunks = []
            for k in range(len(dist)):
                idx = torch.randperm(
                    dist[k].shape[0],
                    device=dist[k].device,
                    generator=generator,
                )[:bs]
                chunks.append(dist[k].index_select(0, idx))

            plans = []
            for k in range(len(dist) - 1):
                src = self._push(chunks[k], k, push)
                plans.append(self.plan(src, chunks[k + 1], masses[k], masses[k + 1]))

            self.batches_.append((chunks, plans))

        self.batch_size_ = bs
        self.masses_ = masses
        return self

    def _path_masses(self, plans, traj, ntraj):
        """Masses of the sampled paths: snapshot-size ratios, as in ``pairwise_OT``."""
        return (self.masses_[:, None] / self.masses_[0]).expand(-1, ntraj).clone()

    def sample(self, n_paths=None, generator=None):
        if not hasattr(self, "batches_"):
            raise RuntimeError("Call fit before sample.")

        if n_paths is None:
            ntraj = self.batch_size_
        else:
            ntraj = -(-n_paths // self.n_batches)  # ceil
        chunks0 = self.batches_[0][0]
        nsnaps = len(chunks0)
        ndim = chunks0[0].shape[1]

        paths = []
        masses = []
        for chunks, plans in self.batches_:
            traj = _sample_trajectory(plans, num_samples=ntraj, generator=generator)
            paths_b = torch.zeros(
                (nsnaps, ntraj, ndim),
                dtype=chunks[0].dtype,
                device=chunks[0].device,
            )
            for k in range(nsnaps):
                paths_b[k] = chunks[k].index_select(0, traj[k])
            paths.append(paths_b)
            masses.append(self._path_masses(plans, traj, ntraj))

        paths = torch.cat(paths, dim=1).permute(1, 0, 2)
        masses = torch.cat(masses, dim=1).permute(1, 0)
        if n_paths is not None:
            paths = paths[:n_paths]
            masses = masses[:n_paths]
        return paths, masses


class Independent(PlanPaths):
    """Independent coupling of the snapshots.

    Every node is drawn uniformly (with replacement) in its snapshot, independently
    across snapshots (product of the empirical marginals). ``push`` plays no role.
    """

    def __init__(self):
        super().__init__(n_batches=1)

    def plan(self, src, tgt, mass_src, mass_tgt):
        n_src = src.shape[0]
        n_tgt = tgt.shape[0]
        return torch.full(
            (n_src, n_tgt),
            1.0 / (n_src * n_tgt),
            dtype=src.dtype,
            device=src.device,
        )

    def fit(self, dist, push=None, generator=None):
        if self._is_laws(dist):
            self.laws_ = [(torch.as_tensor(m), torch.as_tensor(c)) for m, c in dist]
            self.dist_ = None
            return self
        self.laws_ = None
        self.dist_ = self._check(dist)
        return self

    def moments(self, n_paths=None, generator=None):
        if getattr(self, "laws_", None) is None:
            return super().moments(n_paths, generator=generator)
        means = torch.stack([m for m, _ in self.laws_])
        nsnaps, ndim = means.shape
        covs = torch.zeros((nsnaps, nsnaps, ndim, ndim), dtype=means.dtype, device=means.device)
        for k, (_, cov) in enumerate(self.laws_):
            covs[k, k] = cov
        return means, covs

    def sample(self, n_paths=None, generator=None):
        if getattr(self, "dist_", None) is None:
            raise RuntimeError("Call fit on samples before sample.")

        if n_paths is None:
            ntraj = min(snap.shape[0] for snap in self.dist_)
        else:
            ntraj = n_paths
        idx = [
            torch.randint(0, snap.shape[0], (ntraj,), device=snap.device, generator=generator)
            for snap in self.dist_
        ]
        paths = torch.stack(
            [snap.index_select(0, i) for snap, i in zip(self.dist_, idx)],
            dim=1,
        )
        masses = torch.ones(paths.shape[:2], dtype=paths.dtype, device=paths.device)
        return paths, masses


class ExactOT(PlanPaths):
    """Exact OT plans (POT ``emd``), squared Euclidean cost.

    Parameters
    ----------
    n_batches : int, default=1
        Number of mini-batches.
    """

    def plan(self, src, tgt, mass_src, mass_tgt):
        return _compute_pairwise_OT_plan(
            src,
            tgt,
            mass_src,
            mass_tgt,
            reg_m=None,
            method="exact",
        )


class Sinkhorn(PlanPaths):
    """Entropic OT plans (POT ``sinkhorn``), squared Euclidean cost.

    Parameters
    ----------
    reg : float, default=0.1
        Entropic regularization (on the squared Euclidean cost, as in ``pairwise_OT``).
    method : str, default='sinkhorn'
        POT backend name.
    n_batches : int, default=1
        Number of mini-batches.
    """

    def __init__(self, reg=0.1, method="sinkhorn", n_batches=1):
        super().__init__(n_batches=n_batches)
        self.reg = reg
        self.method = method

    def plan(self, src, tgt, mass_src, mass_tgt):
        return _compute_pairwise_OT_plan(
            src,
            tgt,
            mass_src,
            mass_tgt,
            reg=self.reg,
            reg_m=None,
            method=self.method,
        )


class UnbalancedSinkhorn(Sinkhorn):
    """Unbalanced entropic plans (POT ``sinkhorn_unbalanced``).

    Snapshot masses are proportional to the snapshot sizes, and the paths carry masses
    computed from the plans as in ``pairwise_OT`` with ``reg_m``.

    Parameters
    ----------
    reg : float, default=0.1
        Entropic regularization.
    reg_m : float, default=10.0
        Marginal relaxation.
    method : str, default='sinkhorn'
        POT backend name.
    n_batches : int, default=1
        Number of mini-batches.
    """

    weighted = True

    def __init__(self, reg=0.1, reg_m=10.0, method="sinkhorn", n_batches=1):
        super().__init__(reg=reg, method=method, n_batches=n_batches)
        self.reg_m = reg_m

    def _snapshot_masses(self, dist):
        return torch.as_tensor(
            [snap.shape[0] for snap in dist],
            dtype=dist[0].dtype,
            device=dist[0].device,
        )

    def plan(self, src, tgt, mass_src, mass_tgt):
        return _compute_pairwise_OT_plan(
            src,
            tgt,
            mass_src,
            mass_tgt,
            reg=self.reg,
            reg_m=self.reg_m,
            method=self.method,
        )

    def _path_masses(self, plans, traj, ntraj):
        # same computation as pairwise_OT with reg_m, kept identical (normalization included)
        nsnaps = len(traj)
        dtype = plans[0].dtype
        device = plans[0].device

        psi = torch.zeros((nsnaps, ntraj), dtype=dtype, device=device)
        probs = torch.eye(ntraj, dtype=dtype, device=device)
        for k in range(nsnaps):
            if k > 0:
                gamma = plans[k - 1]
                probs = probs @ (gamma / gamma.sum(dim=1, keepdim=True))
            denom = probs.mean(dim=0).index_select(0, traj[k])
            psi[k] = self.masses_[k] / self.batch_size_ / denom
        return psi


# ----------------------------------------------------------------------------------------
# Transport maps: deterministic paths of a moving node cloud
# ----------------------------------------------------------------------------------------

class MapPaths(PathEnsemble):
    """Deterministic paths of a moving cloud of nodes.

    The nodes start at cells of the first snapshot. On every interval the node cloud is
    pushed by the reference dynamics, then moved toward the next snapshot by
    :meth:`step`.

    Parameters
    ----------
    n_paths : int or None, default=None
        Number of paths, with start cells drawn without replacement in the first
        snapshot. If ``None``, all its cells.

    Attributes
    ----------
    paths_ : torch.Tensor of shape (n_paths, nsnaps, ndim)
        Node positions of every path.
    """

    def __init__(self, n_paths=None):
        self.n_paths = n_paths

    @abstractmethod
    def step(self, src, tgt, k):
        """Move the pushed node cloud toward the next snapshot.

        Parameters
        ----------
        src : torch.Tensor of shape (n_paths, ndim)
            Node cloud pushed to ``t_{k+1}``.
        tgt : torch.Tensor of shape (n_tgt, ndim)
            Cells of snapshot k + 1.
        k : int
            Interval index.

        Returns
        -------
        torch.Tensor of shape (n_paths, ndim)
            New node positions at ``t_{k+1}``.
        """

    def fit(self, dist, push=None, generator=None):
        dist = self._check(dist)

        z = dist[0]
        if self.n_paths is not None and self.n_paths < z.shape[0]:
            idx = torch.randperm(z.shape[0], device=z.device, generator=generator)
            z = z.index_select(0, idx[: self.n_paths])

        nodes = [z]
        for k in range(len(dist) - 1):
            z = self.step(self._push(z, k, push), dist[k + 1], k)
            nodes.append(z)

        self.paths_ = torch.stack(nodes, dim=1)
        return self

    def sample(self, n_paths=None, generator=None):
        if not hasattr(self, "paths_"):
            raise RuntimeError("Call fit before sample.")

        n = self.paths_.shape[0]
        device = self.paths_.device
        if n_paths is None or n_paths == n:
            paths = self.paths_
        elif n_paths < n:
            idx = torch.randperm(n, device=device, generator=generator)[:n_paths]
            paths = self.paths_.index_select(0, idx)
        else:
            idx = torch.randint(0, n, (n_paths,), device=device, generator=generator)
            paths = self.paths_.index_select(0, idx)

        masses = torch.ones(paths.shape[:2], dtype=paths.dtype, device=paths.device)
        return paths, masses


class DebiasedSinkhorn(MapPaths):
    """Debiased Sinkhorn map computed on the node cloud itself.

    On interval k, with the pushed nodes ``q`` and the cells ``y`` of the next snapshot,
    the entropic plans ``P_ab: q -> y`` and ``P_aa: q -> q`` (cost ``|x - y|^2 / 2``,
    regularization ``reg``, uniform weights) give every node its debiased displacement,
    read on the cloud's own points::

        z_i = q_i + sum_j n P_ab[i, j] y_j - sum_m n P_aa[i, m] q_m

    The step vanishes when the pushed cloud and the next snapshot have the same law, so
    the true dynamics is a fixed point of the iterated construction. The Sinkhorn
    potentials are kept between calls to :meth:`fit`, to warm-start the next round of an
    iteration.

    Parameters
    ----------
    reg : float, default=0.5
        Entropic regularization in data units, on the cost ``|x - y|^2 / 2``. Should be
        at least a few times the squared nearest-neighbour distance of the snapshots.
    stop_thr : float or None, default=None
        Sinkhorn marginal tolerance. If ``None``, 1e-8 in float64 and 1e-5 in lower
        precision (1e-8 is not reachable in float32).
    method : str, default='sinkhorn_log'
        POT backend name.
    warm_start : bool, default=True
        Start every Sinkhorn solve from the potentials of the previous one.
    n_paths : int or None, default=None
        Number of paths (see :class:`MapPaths`).
    """

    def __init__(
        self,
        reg=0.5,
        stop_thr=None,
        method="sinkhorn_log",
        warm_start=True,
        n_paths=None,
    ):
        super().__init__(n_paths=n_paths)
        self.reg = reg
        self.stop_thr = stop_thr
        self.method = method
        self.warm_start = warm_start
        self.state_ = {}

    def _plan(self, x, y, key):
        """Entropic plan between uniform clouds ``x`` and ``y`` (rows sum to 1 / len(x))."""
        a = torch.full((x.shape[0],), 1.0 / x.shape[0], dtype=x.dtype, device=x.device)
        b = torch.full((y.shape[0],), 1.0 / y.shape[0], dtype=y.dtype, device=y.device)
        cost = ot.dist(x, y, metric="sqeuclidean") / 2
        cost_max = float(cost.max())

        if self.stop_thr is not None:
            stop_thr = self.stop_thr
        elif x.dtype == torch.float64:
            stop_thr = 1e-8
        else:
            stop_thr = 1e-5

        plan, log = ot.sinkhorn(
            a,
            b,
            cost / cost_max,
            self.reg / cost_max,
            method=self.method,
            numItermax=1000000,
            stopThr=stop_thr,
            log=True,
            warn=False,
            warmstart=self.state_.get(key) if self.warm_start else None,
        )

        if self.warm_start:
            if "log_u" in log:
                self.state_[key] = (log["log_u"], log["log_v"])
            else:
                self.state_[key] = (torch.log(log["u"]), torch.log(log["v"]))
        return plan

    def step(self, src, tgt, k):
        n = src.shape[0]
        plan_ab = self._plan(src, tgt, ("ab", k))
        plan_aa = self._plan(src, src, ("aa", k))
        return src + n * (plan_ab @ tgt) - n * (plan_aa @ src)


# ----------------------------------------------------------------------------------------
# Gaussian maps: closed-form maps between Gaussian fits, or between exact Gaussian laws
# ----------------------------------------------------------------------------------------

def _sqrtm_sym(mat):
    """Square root of a symmetric positive semi-definite matrix."""
    vals, vecs = torch.linalg.eigh(mat)
    return (vecs * vals.clamp_min(0).sqrt()) @ vecs.T


def gaussian_ot_matrix(cov_src, cov_tgt):
    """Linear part of the OT map from ``N(., cov_src)`` to ``N(., cov_tgt)``.

    ``T = A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}``, with ``A = cov_src`` and
    ``B = cov_tgt``; ``T A T^T = B``.
    """
    root = _sqrtm_sym(cov_src)
    inv_root = torch.linalg.inv(root)
    return inv_root @ _sqrtm_sym(root @ cov_tgt @ root) @ inv_root


def entropic_map_matrix(cov_src, cov_tgt, eps):
    """Linear part of the entropic barycentric map between two Gaussians.

    Entropic plan between ``N(a, A)`` and ``N(b, B)`` with cost ``|x - y|^2 / 2`` and
    regularization ``eps`` (Janati et al. 2020): ``E[y | x] = b + K (x - a)`` with
    ``K = C^T A^{-1}`` and ``C = A^{1/2} (4 A^{1/2} B A^{1/2} + eps^2 I)^{1/2} A^{-1/2} / 2
    - eps I / 2``.
    """
    eye = torch.eye(cov_src.shape[0], dtype=cov_src.dtype, device=cov_src.device)
    root = _sqrtm_sym(cov_src)
    middle = _sqrtm_sym(4 * root @ cov_tgt @ root + eps**2 * eye)
    cross = root @ middle @ torch.linalg.inv(root) / 2 - eps / 2 * eye
    return cross.T @ torch.linalg.inv(cov_src)


class _GaussianMapPaths(MapPaths):
    """Map paths whose step is linear in the deviation from the mean.

    On interval k the step sends the pushed cloud, of mean ``m_q`` and covariance ``S_q``,
    to ``z = m_{k+1} + L (q - m_q)``, with ``L`` given by :meth:`step_matrix` from ``S_q``
    and the covariance of snapshot k + 1. On samples, the means and covariances are the
    sample moments; on exact laws they are the exact ones, the paths are Gaussian, and
    :meth:`moments` returns their exact moments. With laws, ``push`` must be affine (as
    for a linear model with an affine score): it is applied to ``d + 1`` probe points.
    """

    @abstractmethod
    def step_matrix(self, cov_src, cov_tgt):
        """Linear part ``L`` of the step, from the source and target covariances."""

    def step(self, src, tgt, k):
        mean_src = src.mean(dim=0)
        mean_tgt = tgt.mean(dim=0)
        cov_src = torch.cov(src.T, correction=0)
        cov_tgt = torch.cov(tgt.T, correction=0)
        lin = self.step_matrix(cov_src.double(), cov_tgt.double()).to(src.dtype)
        return mean_tgt + (src - mean_src) @ lin.T

    def fit(self, dist, push=None, generator=None):
        if not self._is_laws(dist):
            self.laws_ = None
            return super().fit(dist, push=push, generator=generator)

        laws = [(torch.as_tensor(m).double(), torch.as_tensor(c).double()) for m, c in dist]
        ndim = laws[0][0].shape[0]
        eye = torch.eye(ndim, dtype=torch.float64)
        mean0, cov0 = laws[0]
        maps = [eye]                                   # x_k - mu_k = G_k (x_0 - m_0)
        means = [mean0]
        for k in range(len(laws) - 1):
            # the step sends the pushed mean to m_{k+1}: only the linear part of the push
            # (its Jacobian, exact from d + 1 probe points for an affine push) matters
            if push is None:
                jac = eye
            else:
                probes = torch.cat([means[-1][None], means[-1][None] + eye], dim=0)
                pushed = self._push(probes, k, push).double()
                jac = (pushed[1:] - pushed[0]).T
            cov_q = jac @ maps[-1] @ cov0 @ maps[-1].T @ jac.T
            lin = self.step_matrix(cov_q, laws[k + 1][1])
            maps.append(lin @ jac @ maps[-1])
            means.append(laws[k + 1][0])

        self.laws_ = laws
        self.means_ = torch.stack(means)
        self.maps_ = torch.stack(maps)
        return self

    def moments(self, n_paths=None, generator=None):
        if getattr(self, "laws_", None) is None:
            return super().moments(n_paths, generator=generator)
        cov0 = self.laws_[0][1]
        covs = torch.einsum("jab,bc,kdc->jkad", self.maps_, cov0, self.maps_)
        return self.means_, covs

    def sample(self, n_paths=None, generator=None):
        if getattr(self, "laws_", None) is None:
            return super().sample(n_paths, generator=generator)
        if n_paths is None:
            raise ValueError("n_paths is required to sample paths fitted on exact laws.")
        mean0, cov0 = self.laws_[0]
        noise = torch.randn(n_paths, mean0.shape[0], dtype=torch.float64, generator=generator)
        x0 = noise @ torch.linalg.cholesky(cov0).T
        paths = self.means_[None] + torch.einsum("kab,nb->nka", self.maps_, x0)
        masses = torch.ones(paths.shape[:2], dtype=paths.dtype)
        return paths, masses


class GaussianOT(_GaussianMapPaths):
    """Exact OT coupling of the Gaussian fits, applied as a map.

    On interval k the pushed node cloud is moved by the OT map between its Gaussian fit
    and that of snapshot k + 1: the nodes reproduce each snapshot's mean and covariance
    exactly, but not its cells. With exact Gaussian laws this is the exact OT coupling of
    the snapshots (the ``n = infinity`` OT coupling).

    Parameters
    ----------
    n_paths : int or None, default=None
        Number of paths (see :class:`MapPaths`).
    """

    def step_matrix(self, cov_src, cov_tgt):
        return gaussian_ot_matrix(cov_src, cov_tgt)


class GaussianDebiasedSinkhorn(_GaussianMapPaths):
    """Debiased Sinkhorn map between Gaussian fits, computed on the node cloud.

    Closed-form counterpart of :class:`DebiasedSinkhorn`: with ``K(A, B)`` the linear part
    of the entropic barycentric map (:func:`entropic_map_matrix`), the step is
    ``z = m_{k+1} + (I + K(S_q, S_{k+1}) - K(S_q, S_q)) (q - m_q)``. With exact Gaussian
    laws this is the ``n = infinity`` node-cloud debiasing.

    Parameters
    ----------
    reg : float, default=0.5
        Entropic regularization in data units, on the cost ``|x - y|^2 / 2``.
    n_paths : int or None, default=None
        Number of paths (see :class:`MapPaths`).
    """

    def __init__(self, reg=0.5, n_paths=None):
        super().__init__(n_paths=n_paths)
        self.reg = reg

    def step_matrix(self, cov_src, cov_tgt):
        eye = torch.eye(cov_src.shape[0], dtype=cov_src.dtype, device=cov_src.device)
        cross = entropic_map_matrix(cov_src, cov_tgt, self.reg)
        self_map = entropic_map_matrix(cov_src, cov_src, self.reg)
        return eye + cross - self_map


__all__ = [
    "PathEnsemble",
    "PlanPaths",
    "MapPaths",
    "Independent",
    "ExactOT",
    "Sinkhorn",
    "UnbalancedSinkhorn",
    "DebiasedSinkhorn",
    "GaussianOT",
    "GaussianDebiasedSinkhorn",
]
