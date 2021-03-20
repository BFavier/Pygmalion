import torch
from typing import Union


def load_batchnorm(dump):
    """returns the BatchNorm1d or BatchNorm2d object loaded from a dump"""
    return globals()[dump["type"]].from_dump(dump)


class BatchNorm:
    """A template for batch norm layers"""

    @classmethod
    def from_dump(cls, dump: dict) -> 'BatchNorm':
        """returns a 'BatchNorm' layer from a dump"""
        cls = globals()[dump["type"]]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.num_features = dump["num features"]
        obj.momentum = dump["momentum"]
        obj.affine = dump["affine"]
        obj.eps = dump["eps"]
        obj.track_running_stats = dump["track running stats"]
        obj.register_buffer("num_batches_tracked",
                            torch.tensor(dump["num batches tracked"],
                                         dtype=torch.long))
        obj.register_buffer("running_mean",
                            torch.tensor(dump["running mean"],
                                         dtype=torch.float))
        obj.register_buffer("running_var",
                            torch.tensor(dump["running var"],
                                         dtype=torch.float))
        if obj.affine:
            obj.weight = torch.nn.Parameter(torch.tensor(dump["weight"],
                                                         dtype=torch.float))
            obj.bias = torch.nn.Parameter(torch.tensor(dump["bias"],
                                                       dtype=torch.float))
        else:
            obj.weight = None
            obj.bias = None
        return obj

    @property
    def dump(self) -> dict:
        w = self.weight.tolist() if self.affine else None
        b = self.bias.tolist() if self.affine else None
        return {"type": type(self).__name__,
                "num features": self.num_features,
                "running mean": self.running_mean.tolist(),
                "running var": self.running_var.tolist(),
                "track running stats": self.track_running_stats,
                "num batches tracked": self.num_batches_tracked.tolist(),
                "momentum": self.momentum,
                "eps": self.eps,
                "affine": self.affine,
                "weight": w,
                "bias": b}

    def undo(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse transform to a tensor

        Parameters
        ----------
        X : torch.Tensor
            the normalized tensor

        Returns
        -------
        torch.Tensor :
            The un-normalized tensor
        """
        X = torch.transpose(X, 1, -1)
        Y = X*(self.running_var+self.eps)**0.5 + self.running_mean
        return torch.transpose(Y, -1, 1)


class BatchNorm1d(torch.nn.BatchNorm1d, BatchNorm):
    """A wrapper around torch.nn.BatchNorm1d"""

    def __init__(self, num_features: int,
                 momentum: Union[float, None] = 0.1,
                 affine: bool = False,
                 eps: float = 1.0E-5):
        """
        Parameters
        ----------
        num_features : int
            the number of channels to normalize
        momentum : float of None
            the update factor for the running mean and variance:
            x_run = x_run*(1-momentum) + x_new*momentum
            if None, the cumulative average and variance are calculated instead
        affine : bool
            whether to add learnable weight and bias after normalization
        eps : float
            A small constant used to avoid dividing by 0 when normalizing
        """
        torch.nn.BatchNorm1d.__init__(self, num_features,
                                      momentum=momentum,
                                      affine=affine,
                                      eps=eps)


class BatchNorm0d(BatchNorm1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BatchNorm2d(torch.nn.BatchNorm2d, BatchNorm):
    """A wrapper around torch.nn.BatchNorm2d"""

    def __init__(self, num_features: int,
                 momentum: Union[float, None] = 0.1,
                 affine: bool = False,
                 eps: float = 1.0E-5):
        """
        Parameters
        ----------
        num_features : int
            the number of channels to normalize
        momentum : float of None
            the update factor for the running mean and variance:
            x_run = x_run*(1-momentum) + x_new*momentum
            if None, the cumulative average and variance are calculated instead
        affine : bool
            whether to add learnable weight and bias after normalization
        eps : float
            A small constant used to avoid dividing by 0 when normalizing
        """
        torch.nn.BatchNorm2d.__init__(self, num_features,
                                      momentum=momentum,
                                      affine=affine,
                                      eps=eps)
