import torch
import torch.nn.functional as F
from typing import Union


class Dropout(torch.nn.Module):
    """
    In training mode, randomly zero out each channel with probability 'p'
    (over the whole feature map if in 1d/2d)
    This is usefull to enforce redundancy in the channels of the neural
    network, thus increasing it's capacity to extrapolate to unseen data
    """

    @classmethod
    def from_dump(cls, dump: dict) -> "Dropout":
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.p = dump["p"]

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "p": self.p}

    def __init__(self, p: Union[None, float]):
        super().__init__()
        self.p = p

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.p is None:
            return X
        return F.dropout2d(X, self.p, training=self.training)
