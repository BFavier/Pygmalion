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
        self.running_mean = torch.nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype), requires_grad=False)
        self.running_var = torch.nn.Parameter(torch.ones(num_features, device=device, dtype=dtype), requires_grad=False)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor]=None, track_running_stats: bool=True) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (N, ..., num_features, ...)
        mask : torch.Tensor or None
            tensor of boolean of shape (N, ...) (same shape as X except missing the 'dim' dimension).
            Masked values are ignored in calculation of mean and variance
        
        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, ..., num_features, ...) normalized along the given dimension
        """
        X = X.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if X.shape[self.dim] != self.num_features:
            raise ValueError(f"Expected {self.num_features} size for dimension {self.dim} but got tensor of shape {tuple(X.shape)}")
        if self.training and track_running_stats:
            with torch.no_grad():
                Xr = X.moveaxis(self.dim, 0).reshape(self.num_features, -1)
                if mask is not None:
                    Xr = Xr * ~mask.to(Xr.device).reshape(-1).unsqueeze(0)
                n = Xr.shape[-1] if mask is None else Xr.shape[-1] - mask.sum()
                mean = Xr.sum(dim=-1) / max(1, n)
                var = torch.sum((Xr - mean.unsqueeze(-1))**2, dim=-1) / max(1, n)
                self.running_var.data = (self.n_observations/(self.n_observations+n)) * self.running_var + (n/(self.n_observations+n)) * var + self.n_observations*n/(self.n_observations+n)**2 * (mean - self.running_mean)**2
                self.running_mean.data = self.running_mean * (self.n_observations / (self.n_observations + n)) + mean * (n / (self.n_observations + n))
                self.n_observations += n
        shape = [self.num_features if i == self.dim % len(X.shape) else 1 for i, _ in enumerate(X.shape)]
        X = (X - self.running_mean.reshape(shape)) / (self.running_var.reshape(shape) + self.eps)**0.5
        return X

    def unscale(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Unapply normalization
        """
        shape = [self.num_features if i == self.dim % len(Y.shape) else 1 for i, _ in enumerate(Y.shape)]
        return Y * (self.running_var.reshape(shape) + self.eps)**0.5 + self.running_mean.reshape(shape)

    @property
    def device(self) -> torch.device:
        return self.running_mean.device
