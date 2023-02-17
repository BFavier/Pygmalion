import torch
import torch.nn.functional as F
from typing import Union, Callable


class Activation(torch.nn.Module):
    """
    creates a wrapper torch.nn.Module around an activation function
    provided by the user

    Example
    -------
    >>> Activation("relu")

    """

    def __new__(cls, activation: Union[str, Callable, torch.nn.Module],
                *args, **kwargs) -> torch.nn.Module:
        if isinstance(activation, torch.nn.Module):
            return activation
        else:
            obj = super().__new__(cls)
            cls.__init__(obj, activation, *args, **kwargs)
            return obj

    def __init__(self, activation: Union[str, Callable]):
        super().__init__()
        assert isinstance(activation, str) or callable(activation)
        self.function = self._as_callable(activation)

    def __repr__(self):
        return f"Activation({self.function.__name__})"

    def forward(self, X):
        return self.function(X)

    def _as_callable(self, activation: Union[str, Callable]):
        if isinstance(activation, str):
            if hasattr(torch, activation):
                return getattr(torch, activation)
            elif hasattr(F, activation):
                return getattr(F, activation)
            else:
                ValueError(f"Unknown pytorch function '{activation}'")
        else:
            return activation
