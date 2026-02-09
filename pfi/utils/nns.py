"""Neural-network building blocks and differential operators for PFI."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.autograd import grad


class _CompoundModel(nn.Module):
    """Common container model with optional scale propagation to `net`."""

    def __init__(
        self,
        net,
    ):
        super(_CompoundModel, self).__init__()
        self.net = net

    def set_scales(
        self,
        mean,
        std,
    ):
        if hasattr(self.net, "set_scales"):
            self.net.set_scales(mean, std)
        return self


class BatchNorm(object):
    """Simple affine normalizer ``(x - mean) / std``.

    Parameters
    ----------
    mean : float or torch.Tensor of shape (n_features,) or (1, n_features)
        Feature-wise mean.
    std : float or torch.Tensor of shape (n_features,) or (1, n_features)
        Feature-wise standard deviation.
    """

    def __init__(
        self,
        mean,
        std,
    ):
        self.mean = mean
        self.std = std

    def __call__(
        self,
        x,
    ):
        """Normalize an input tensor.

        Parameters
        ----------
        x : torch.Tensor of shape (..., n_features)
            Input tensor.

        Returns
        -------
        x_norm : torch.Tensor of shape (..., n_features)
            Normalized tensor.
        """
        return (x - self.mean) / self.std


class LayerNoWN(nn.Module):
    """Linear layer with Xavier initialization and no weight normalization.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    seed : int
        Random seed used for initialization.
    activation : torch.nn.Module
        Activation used in surrounding network to set initialization gain.

    """

    def __init__(
        self,
        in_features,
        out_features,
        seed,
        activation,
    ):
        super(LayerNoWN, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        gain = 5 / 3 if isinstance(activation, nn.Tanh) else 1
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x,
    ):
        """Apply the linear map.

        Parameters
        ----------
        x : torch.Tensor of shape (..., in_features)
            Input tensor.

        Returns
        -------
        y : torch.Tensor of shape (..., out_features)
            Output tensor.
        """
        return self.linear(x)


class DNN(nn.Module):
    """Fully-connected network with optional feature normalization.

    Parameters
    ----------
    sizes : list of int
        Layer sizes including input and output dimensions.
    mean : float or torch.Tensor, default=0
        Initial mean used by the internal normalizer.
    std : float or torch.Tensor, default=1
        Initial standard deviation used by the internal normalizer.
    seed : int, default=0
        Random seed for layer initialization.
    activation : torch.nn.Module, default=torch.nn.Tanh()
        Hidden activation module.

    """

    def __init__(
        self,
        sizes,
        mean=0,
        std=1,
        seed=0,
        activation=nn.Tanh(),
    ):
        super(DNN, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.bn = BatchNorm(mean, std)
        layer = []
        for i in range(len(sizes) - 2):
            linear = LayerNoWN(sizes[i], sizes[i + 1], seed, activation)
            layer += [linear, activation]
        layer += [LayerNoWN(sizes[-2], sizes[-1], seed, activation)]
        self.net = nn.Sequential(*layer)

    def set_scales(self, mean, std):
        """Update normalization statistics.

        Parameters
        ----------
        mean : torch.Tensor of shape (1, n_features)
            New feature means.
        std : torch.Tensor of shape (1, n_features)
            New feature standard deviations.

        Returns
        -------
        self : DNN
            Estimator instance.
        """
        self.bn = BatchNorm(mean, std)
        return self

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
        return self.net(self.bn(x))


class SpectralNormDNN(nn.Module):
    """Fully-connected network with spectral normalization on hidden layers.

    Parameters
    ----------
    sizes : list of int
        Layer sizes including input and output dimensions.
    mean : float or torch.Tensor, default=0
        Initial mean used by the internal normalizer.
    std : float or torch.Tensor, default=1
        Initial standard deviation used by the internal normalizer.
    seed : int, default=0
        Random seed for layer initialization.
    activation : torch.nn.Module, default=torch.nn.Tanh()
        Hidden activation module.

    """

    def __init__(
        self,
        sizes,
        mean=0,
        std=1,
        seed=0,
        activation=nn.Tanh(),
    ):
        super(SpectralNormDNN, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.bn = BatchNorm(mean, std)
        layers = []
        for i in range(len(sizes) - 2):
            linear = nn.Linear(sizes[i], sizes[i + 1])
            linear = nn_utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(activation)
        final_linear = nn.Linear(sizes[-2], sizes[-1])
        layers.append(final_linear)
        self.net = nn.Sequential(*layers)

    def set_scales(self, mean, std):
        """Update normalization statistics.

        Parameters
        ----------
        mean : torch.Tensor of shape (1, n_features)
            New feature means.
        std : torch.Tensor of shape (1, n_features)
            New feature standard deviations.

        Returns
        -------
        self : SpectralNormDNN
            Estimator instance.
        """
        self.bn = BatchNorm(mean, std)
        return self

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
        return self.net(self.bn(x))

    def set_scales(self, mean, std):
        """Update normalization statistics.

        Parameters
        ----------
        mean : torch.Tensor of shape (1, n_features)
            New feature means.
        std : torch.Tensor of shape (1, n_features)
            New feature standard deviations.

        Returns
        -------
        self : DNN
            Estimator instance.
        """
        self.bn = BatchNorm(mean, std)
        return self
    
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
    ):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
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
            r = torch.randperm(self.dataset_len)
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
):
    """Compute coordinate-wise divergence terms of a vector field.

    Parameters
    ----------
    field : torch.Tensor of shape (batch_size, ndim)
        Vector field evaluated at ``x``.
    x : torch.Tensor of shape (batch_size, ndim)
        Input points with gradient tracking enabled.

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
            create_graph=True,
        )[0]
        div[:, i] = gradient[:, i]
    return div


class FreezeVarDNN(nn.Module):
    """Wrapper that fixes one input feature to a constant before inference.

    Parameters
    ----------
    dnn : torch.nn.Module
        Base network receiving ``n_features`` inputs.
    var_index : int
        Index of the feature to overwrite.
    var_value : float
        Constant value assigned to that feature.

    """

    def __init__(
        self,
        dnn,
        var_index,
        var_value,
    ):
        super(FreezeVarDNN, self).__init__()
        self.dnn = dnn
        self.var_index = var_index
        self.var_value = var_value

    def forward(
        self,
        x,
    ):
        """Evaluate the wrapped network with one fixed input feature.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_features)
            Input batch.

        Returns
        -------
        y : torch.Tensor of shape (batch_size, n_outputs)
            Network prediction.
        """
        left = x[:, : self.var_index]
        right = x[:, self.var_index :]
        frozen = torch.full(
            (x.shape[0], 1),
            self.var_value,
            dtype=x.dtype,
            device=x.device,
        )
        x_frozen = torch.cat((left, frozen, right), dim=1)
        return self.dnn(x_frozen)
