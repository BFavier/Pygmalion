import torch
import torch.nn.functional as F
from typing import Union, Callable
from types import LambdaType


class Activation(torch.nn.Module):
    """
    creates a wrapper torch.nn.Module around an activation function
    provided by the user

    Example
    -------
    >>> Activation("relu")

    """

    def __init__(self, activation: Union[str, Callable]):
        super().__init__()
        assert isinstance(activation, str) or callable(activation)
        assert not callable(activation) or activation.__name__ != "<lambda>", "Lambda function cannot be pickled and saved on disk"
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
