from typing import Optional
import torch


class Normalizer(torch.nn.Module):
    """
    Normalize a tensor along the given dimension based on a running average
    of all training observations. Suitable only to normalize input or target data,
    as normalization parameters will change less and less as more batches are seen.
    """

    def __init__(self, dim: int, num_features: int, eps: float=1e-05,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
        dim : int
            dimension along which data are normalised
        num_features : int
            size of the given dimension along which data are normalized
        eps : float
            epsilon factor to avoid division by zero
        momentum : float
            update factor used for the running stats
        device : torch.device or None
            device to store the parameters and tensors on
        dtype : torch.dtype
            dtype of the tensors and parameters
        """
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.eps = eps
        self.n_observations = 0
        self.running_mean = torch.zeros(num_features, device=device, dtype=dtype)
        self.running_var = torch.ones(num_features, device=device, dtype=dtype)

    def forward(self, X: torch.Tensor, track_running_stats: bool=True) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (N, ..., num_features, ...)
        
        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, ..., num_features, ...) normalized along the given dimension
        """
        if X.shape[self.dim] != self.num_features:
            raise ValueError(f"Expected tensor of shape (N, *, {self.num_features}, *) but got {tuple(X.shape)}")
        if self.training and track_running_stats:
            with torch.no_grad():
                self.running_mean, self.running_var = self.running_mean.to(X.device), self.running_var.to(X.device)
                n = X.numel() // self.num_features
                var = X.moveaxis(self.dim, 0).reshape(self.num_features, -1).var(dim=-1, unbiased=False)
                mean = X.moveaxis(self.dim, 0).reshape(self.num_features, -1).mean(dim=self.dim)
                self.running_var = (self.n_observations/(self.n_observations+n)) * self.running_var + (n/(self.n_observations+n)) * var + self.n_observations*n/(self.n_observations+n)**2 * (mean - self.running_mean)**2
                self.running_mean = mean * (self.n_observations / (self.n_observations + n)) + self.running_mean * (n / (self.n_observations + n))
                self.n_observations += n
        shape = [self.num_features if i == self.dim else 1 for i, _ in enumerate(X.shape)]
        X = (X - self.running_mean.reshape(shape)) / (self.running_var.reshape(shape) + self.eps)**0.5
        return X

    def unscale(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Unapply normalization
        """
        shape = [self.num_features if i == self.dim else 1 for i, _ in enumerate(Y.shape)]
        return Y * (self.running_var.reshape(shape) + self.eps)**0.5 + self.running_mean.reshape(shape)


class FeaturesNorm(torch.nn.Module):
    """
    Similar to layer norm but normalize along any given dimension.
    Performs normalization of each observation along a given dimension,
    and an optional additional affine transform
    """

    def __init__(self, dim: int, num_features: int, eps: float=1e-05, elementwise_affine: bool=True,
                 device: torch.device=None, dtype: torch.dtype=None):
        """
        Parameters
        ----------
        dim : int
            dimension along which to normalize
        num_features : int
            size of the tensors to normalize along the given dimension
        eps : float
            numerical epsilon to avoid division by zero
        elementwise_affine : bool
            whether to apply affine transform in addition to normalization
        device : torch.device
            device to store the parameters one
        dtype : torch.dtype
            data type of the parameters
        """
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.eps = eps
        self.weight = torch.nn.parameter.Parameter(torch.ones(num_features, device=device, dtype=dtype)) if elementwise_affine else None
        self.bias = torch.nn.parameter.Parameter(torch.zeros(num_features, device=device, dtype=dtype)) if elementwise_affine else None
    
    def forward(self, X: torch.Tensor):
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (N, ..., num_features, ...)
        
        Returns
        -------
        torch.Tensor :
            tensor of same shape normalized (and affine transformed) along 'dim'
        """
        if X.shape[self.dim] != self.num_features:
            raise ValueError(f"Expected tensor of shape (N, *, {self.num_features}, *) but got {tuple(X.shape)}")
        X = (X - torch.mean(X, dim=self.dim).unsqueeze(self.dim))/(torch.std(X, dim=self.dim, unbiased=False).unsqueeze(self.dim) + self.eps)
        shape = [self.num_features if i == self.dim else 1 for i, _ in enumerate(X.shape)]
        if self.weight is not None:
            X = X * self.weight.reshape(shape)
        if self.bias is not None:
            X = X + self.bias.reshape(shape)
        return X