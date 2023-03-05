from typing import Optional
import torch


class Normalizer(torch.nn.Module):

    def __init__(self, num_features: int, eps: float=1e-05, momentum: float=0.1,
                 affine: bool=True, device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(num_features, device=device, dtype=dtype)
        self.running_var = torch.ones(num_features, device=device, dtype=dtype)
        if affine:
            self.weight = torch.nn.parameter.Parameter(torch.ones(num_features, device=device, dtype=dtype))
            self.bias = torch.nn.parameter.Parameter(torch.ones(num_features, device=device, dtype=dtype))
        else:
            self.weight, self.bias = (None, None)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (N, D, *)
        
        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, D, *) normalized along 1st dimension
        """
        if X.shape[1] != self.num_features:
            raise ValueError(f"Expected tensor of shape (N, {self.num_features}, *) but got {tuple(X.shape)}")
        if self.training:
            with torch.no_grad():
                x = X.transpose(0, 1).reshape(self.num_features, -1)
                mean = x.mean(dim=1)
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                var = x.var(dim=1, unbiased=False)
                self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        shape = [self.num_features if i == 1 else 1 for i, _ in enumerate(X.shape)]
        X = (X - self.running_mean.reshape(shape)) / (self.running_var.reshape(shape) + self.eps)**0.5
        if self.weight is not None:
            X = X * self.weight.reshape(shape)
        if self.bias is not None:
            X = X + self.bias.reshape(shape)
    
    def unapply(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Unapply normalization
        """
        shape = [self.num_features if i == 1 else 1 for i, _ in enumerate(Y.shape)]
        return Y * (self.running_var.reshape(shape) + self.eps)**0.5 + self.running_mean.reshape(shape)
