import torch
import torch.nn.functional as F


class Activation(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'Activation':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.activation = dump["activation"]
        return obj

    def __init__(self, activation: str):
        torch.nn.Module.__init__(self)
        self.activation = activation

    def forward(self, X):
        return self.function(X)

    @property
    def function(self):
        if hasattr(torch, self.activation):
            return getattr(torch, self.activation)
        else:
            return getattr(F, self.activation)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "activation": self.activation}
