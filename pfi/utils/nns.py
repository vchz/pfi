"""Neural-network building blocks and differential operators for PFI."""

import copy

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.autograd import grad



def make_linear_layer(in_features, out_features, generator=None, init="kaiming", gain=1):
    """Create an ``nn.Linear`` layer with controlled weight initialization.

    Parameters
    ----------
    in_features : int
    out_features : int
    generator : torch.Generator or None, default=None
        Local RNG. If ``None``, skips re-initialization (global RNG default).
    init : {"kaiming", "xavier"}, default="kaiming"
        ``"xavier"`` uses ``xavier_normal_`` + zero bias (for ``DNN``).
        ``"kaiming"`` replicates ``nn.Linear.reset_parameters()`` (for ``SpectralNormDNN``).
    gain : float, default=1
        Gain forwarded to ``xavier_normal_`` when ``init="xavier"``.

    Returns
    -------
    nn.Linear
    """
    layer = nn.Linear(in_features, out_features)
    if generator is None:
        return layer
    dev = generator.device
    with torch.no_grad():
        if init == "xavier":
            std = gain * (2 / (in_features + out_features)) ** 0.5
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator, device=dev) * std)
            layer.bias.zero_()
        else:
            bound = 1 / in_features ** 0.5
            layer.weight.copy_(torch.rand(layer.weight.shape, generator=generator, device=dev) * 2 * bound - bound)
            layer.bias.copy_(torch.rand(layer.bias.shape, generator=generator, device=dev) * 2 * bound - bound)
    return layer


def add_spectral_norm(layer, generator=None):
    """Wrap ``layer`` with spectral normalization and seed the power-iteration vectors.

    ``nn_utils.spectral_norm`` initializes ``weight_u`` and ``weight_v`` from
    the global RNG.  When a local ``generator`` is provided, those buffers are
    overwritten here so that the full initialization is reproducible.

    Parameters
    ----------
    layer : nn.Linear
        Layer to wrap.
    generator : torch.Generator or None, default=None
        Local RNG used to re-initialize ``weight_u`` / ``weight_v``.

    Returns
    -------
    nn.Module
        Spectrally-normalized layer.
    """
    wrapped = nn_utils.spectral_norm(layer)
    if generator is not None:
        with torch.no_grad():
            for name in ("weight_u", "weight_v"):
                buf = getattr(wrapped, name)
                v = torch.randn(buf.shape, generator=generator, dtype=buf.dtype, device=buf.device)
                buf.copy_(v / (v.norm() + 1e-12))
    return wrapped


class DNN(nn.Module):
    """Fully-connected network.

    Parameters
    ----------
    sizes : list of int
        Layer sizes including input and output dimensions.
    activation : torch.nn.Module, default=torch.nn.ELU()
        Hidden activation module.
    generator : torch.Generator or None, default=None
        Optional RNG for reproducible weight initialization.

    """

    def __init__(
        self,
        sizes,
        activation=nn.ELU(),
        generator=None,
    ):
        super(DNN, self).__init__()
        gain = 5 / 3 if isinstance(activation, nn.Tanh) else 1
        layers = []
        for i in range(len(sizes) - 2):
            layers += [make_linear_layer(sizes[i], sizes[i + 1], generator=generator, init="xavier", gain=gain), activation]
        layers += [make_linear_layer(sizes[-2], sizes[-1], generator=generator, init="xavier", gain=gain)]
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x,
    ):
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (..., n_features)
            Input tensor.

        Returns
        -------
        y : torch.Tensor of shape (..., sizes[-1])
            Network output.
        """
        return self.net(x)

class SpectralNormDNN(nn.Module):
    """Fully-connected network with spectral normalization on hidden layers.

    Parameters
    ----------
    sizes : list of int
        Layer sizes including input and output dimensions.
    dim_x: int, default=None
        The dimension of the x variables to apply spectral norm.
        These are always the first dim_x dimensions of the data.
        The last dimensions are assumed to be 'time-like'.
    activation : torch.nn.Module, default=torch.nn.ELU()
        Hidden activation module.
    full : bool, default=False
        If ``True``, also apply spectral normalization to the final layer.
    generator : torch.Generator or None, default=None
        Optional RNG for reproducible weight initialization.

    """
    def __init__(
        self,
        sizes,
        dim_x=None,
        activation=nn.ELU(),
        full=False,
        generator=None,
    ):
        super(SpectralNormDNN, self).__init__()

        self.dim_x = dim_x if dim_x is not None else sizes[0]
        self.has_t = self.dim_x < sizes[0]
        self.activation = activation

        self.linear_x = add_spectral_norm(
            make_linear_layer(self.dim_x, sizes[1], generator=generator, init="kaiming"),
            generator=generator,
        )
        if self.has_t:
            self.linear_t = make_linear_layer(
                sizes[0] - self.dim_x, sizes[1], generator=generator, init="kaiming"
            )

        layers = []
        for i in range(1, len(sizes) - 2):
            layers.append(add_spectral_norm(
                make_linear_layer(sizes[i], sizes[i + 1], generator=generator, init="kaiming"),
                generator=generator,
            ))
            layers.append(activation)

        final = make_linear_layer(sizes[-2], sizes[-1], generator=generator, init="kaiming")
        if full:
            final = add_spectral_norm(final, generator=generator)
        layers.append(final)

        self.rest = nn.Sequential(*layers)

    def forward(self, x):
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (..., n_features)
            Input tensor.

        Returns
        -------
        y : torch.Tensor of shape (..., sizes[-1])
            Network output.
        """
        h = self.linear_x(x[..., :self.dim_x])
        if self.has_t:
            h = h + self.linear_t(x[..., self.dim_x:])
        return self.rest(self.activation(h))
    

class LinearNN(nn.Module):
    """Single linear map parameterized by a matrix ``B``.

    Parameters
    ----------
    sizes : list of int
        Two entries ``[in_dim, out_dim]``.
    activation : torch.nn.Module, default=torch.nn.ELU()
        Kept for API compatibility.

    """

    def __init__(self, sizes, activation=nn.ELU()):
        super().__init__()
        _ = activation
        in_dim = int(sizes[0])
        out_dim = int(sizes[-1])
        self.B = nn.Parameter(torch.zeros(out_dim, in_dim))

    def forward(self, x):
        """Apply the linear map to inputs ``x``.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, in_dim)
            Input tensor.

        Returns
        -------
        torch.Tensor of shape (n_samples, out_dim)
            Linear output.
        """
        return torch.einsum("mr,nr->nm", self.B, x)

class ConstantNN(nn.Module):
    """Constant matrix module parameterized by ``B``.

    Parameters
    ----------
    sizes : list of int
        Two entries ``[in_dim, out_dim]``. The matrix ``B`` has shape
        ``(out_dim, in_dim)``.
    activation : torch.nn.Module, default=torch.nn.ELU()
        Kept for API compatibility.

    """

    def __init__(self, sizes, activation=nn.ELU()):
        super().__init__()
        _ = activation
        in_dim = int(sizes[0])
        out_dim = int(sizes[-1])
        self.B = nn.Parameter(torch.zeros(out_dim, in_dim))

    def forward(self, x):
        """Return the learned constant matrix ``B``.

        Parameters
        ----------
        x : torch.Tensor
            Unused input, kept for API compatibility.

        Returns
        -------
        torch.Tensor of shape (out_dim, in_dim)
            Constant matrix parameter.
        """
        _ = x
        return self.B

    
class FastTensorDataLoader:
    """Lightweight mini-batch iterator over in-memory tensors.

    Parameters
    ----------
    *tensors : tuple of torch.Tensor
        Tensors with matching first dimension.
    batch_size : int, default=32
        Number of samples per yielded batch.
    shuffle : bool, default=False
        If ``True``, shuffle samples at each iteration.

    """

    def __init__(
        self,
        *tensors,
        batch_size=32,
        shuffle=False,
        generator=None,
    ):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(
        self,
    ):
        """Initialize iteration.

        Returns
        -------
        self : FastTensorDataLoader
            Iterator instance.
        """
        if self.shuffle:
            r = torch.randperm(self.dataset_len, 
                               generator=self.generator, 
                               device=self.generator.device)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(
        self,
    ):
        """Return the next batch.

        Returns
        -------
        batch : tuple of torch.Tensor
            Batched slices from input tensors.

        Raises
        ------
        StopIteration
            When the full dataset has been iterated.
        """
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(
        self,
    ):
        """Return the number of batches per epoch.

        Returns
        -------
        n_batches : int
            Number of batches.
        """
        return self.n_batches

def loss_grad_std(
    loss,
    net,
    device,
):
    """Estimate the pooled standard deviation of layer gradients.

    Parameters
    ----------
    loss : torch.Tensor of shape ()
        Scalar loss value.
    net : torch.nn.Module
        Network containing linear layers.
    device : str or torch.device
        Device used for intermediate tensors.

    Returns
    -------
    std : torch.Tensor of shape ()
        Pooled gradient standard deviation across linear layer parameters.
    """
    var = []
    siz = []
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        w = grad(loss, m.weight, retain_graph=True, allow_unused=True)[0]
        b = grad(loss, m.bias, retain_graph=True, allow_unused=True)[0]
        if w is None or b is None:
            continue
        wb = torch.cat((w.view(-1), b))
        nit = torch.numel(wb)
        var.append((nit - 1) * torch.var(wb))
        siz.append(nit)
    vart = torch.tensor(var, dtype=torch.float32, device=device)
    sizt = torch.tensor(siz, dtype=torch.float32, device=device)
    return torch.sqrt(torch.sum(vart) / (torch.sum(sizt) - len(sizt)))

def divergence(
    field,
    x,
    create_graph=True,
):
    """Compute coordinate-wise divergence terms of a vector field.

    Parameters
    ----------
    field : torch.Tensor of shape (batch_size, ndim)
        Vector field evaluated at ``x``.
    x : torch.Tensor of shape (batch_size, ndim)
        Input points with gradient tracking enabled.
    create_graph : bool, default=True
        If ``True``, keep higher-order graph information. Set to ``False``
        for inference-only divergence computation.

    Returns
    -------
    div : torch.Tensor of shape (batch_size, ndim)
        Diagonal Jacobian terms, one per coordinate.
    """
    dim = field.shape[1]
    div = torch.zeros((field.shape[0], dim), device=x.device)
    for i in range(dim):
        out_ = field[:, i]
        gradient = grad(
            outputs=out_,
            inputs=x,
            grad_outputs=torch.ones_like(out_),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        div[:, i] = gradient[:, i]
    return div

def symsqrt(matrix):
    """Compute the symmetric square root of a positive matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        Positive-definite matrix of shape ``(..., n, n)``.

    Returns
    -------
    torch.Tensor
        Symmetric square root of ``matrix`` with the same batch shape.
    """
    # perform the decomposition
    # s, v = matrix.symeig(eigenvectors=True)
    _, s, v = matrix.svd()  # passes torch.autograd.gradcheck()
    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    # compose the square root matrix
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def clone_module(module):
    """Clone a ``torch.nn.Module`` safely, sharing registered sub-module references.

    Standard ``deepcopy`` fails when a module contains non-leaf tensors, for
    example the ``weight_u`` / ``weight_v`` buffers written by spectral-norm
    during a forward pass.  This function works around the issue by temporarily
    detaching all registered sub-modules (``module._modules``) from the module
    registry, deepcopying only the skeleton (buffers and plain attributes, which
    are always leaf tensors), and then re-attaching the sub-modules by shared
    reference on both the original and the clone.

    Sharing sub-module references is correct for our use case: the sub-modules
    attached to a flow model (e.g. a pre-trained score model) are never modified
    during flow training.

    Parameters
    ----------
    module : torch.nn.Module
        Module to clone.

    Returns
    -------
    torch.nn.Module
        Cloned module with the same skeleton and shared sub-module references.
    """
    saved = {}
    for name in list(module._modules.keys()):
        saved[name] = module._modules.pop(name)

    try:
        cloned = copy.deepcopy(module)
    finally:
        for name, submod in saved.items():
            setattr(module, name, submod)

    for name, submod in saved.items():
        setattr(cloned, name, submod)

    return cloned
