import torch
from typing import Union


def load_batchnorm(dump):
    """returns the BatchNorm1d or BatchNorm2d object loaded from a dump"""
    return globals()[dump["type"]].from_dump(dump)


class _BatchNorm:
    """A template for batch norm layers"""

    @classmethod
    def from_dump(cls, dump: dict) -> '_BatchNorm':
        """returns a '_BatchNorm' layer from a dump"""
        assert dump["type"] == cls.__name__
        obj = cls(dump["num features"],
                  momentum=dump["momentum"],
                  affine=dump["affine"],
                  eps=dump["eps"])
        obj.running_mean = torch.tensor(dump["running mean"],
                                        dtype=torch.float)
        obj.running_var = torch.tensor(dump["running var"],
                                       dtype=torch.float)
        if obj.affine:
            obj.weight = torch.nn.Parameter(torch.tensor(dump["weight"],
                                                         dtype=torch.float))
            obj.bias = torch.nn.Parameter(torch.tensor(dump["bias"],
                                                       dtype=torch.float))
        return obj

    @property
    def dump(self) -> dict:
        w = self.weight.tolist() if self.affine else None
        b = self.bias.tolist() if self.affine else None
        return {"type": type(self).__name__,
                "num features": self.num_features,
                "running mean": self.running_mean.tolist(),
                "running var": self.running_var.tolist(),
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
        return X*(self.running_var+self.eps)**0.5 + self.running_mean


class BatchNorm1d(torch.nn.BatchNorm1d, _BatchNorm):
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


class BatchNorm2d(torch.nn.BatchNorm2d, _BatchNorm):
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
