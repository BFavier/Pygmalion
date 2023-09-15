import torch


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
        shape = [self.num_features if i == self.dim % len(X.shape) else 1 for i, _ in enumerate(X.shape)]
        if self.weight is not None:
            X = X * self.weight.reshape(shape)
        if self.bias is not None:
            X = X + self.bias.reshape(shape)
        return X