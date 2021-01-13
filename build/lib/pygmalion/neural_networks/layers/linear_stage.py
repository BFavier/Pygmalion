import torch
from .linear import Linear
from .activation import Activation
from .batch_norm import BatchNorm1d


class LinearStage(torch.nn.Module):
    """
    An 'LinearStage' module is a 'Linear' layer followed by an
    'Activation' layer and finally a 'BatchNorm1d' layer.

    Attributes
    ----------
    linear : Linear
        The linear layer
    activation : Activation
        The activation layer
    batch_norm : BatchNorm1d
        The batch normalization layer
    """

    @classmethod
    def from_dump(cls, dump: dict) -> 'LinearStage':
        """returns a 'LinearStage' obStageject from a dump"""
        assert dump["type"] == cls.__name__
        obj = cls(1, 1)
        obj.linear = Linear.from_dump(dump["linear"])
        obj.activation = Activation.from_dump(dump["activation"])
        obj.batch_norm = BatchNorm1d.from_dump(dump["batch norm"])
        return obj

    def __init__(self, in_features: int,
                 out_features: int = 8,
                 activation: str = "relu"):
        """
        Parameters
        ----------
        in_features : int
            The nuber of features in the input
        out_features : int
            The number of features in the ouput
        activation : str
            The name of the activation layer
        """
        super().__init__()
        self.linear = Linear(in_features, out_features)
        self.activation = Activation(activation)
        self.batch_norm = BatchNorm1d(out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(self.activation(self.linear(input)))

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "linear": self.linear.dump,
                "activation": self.activation.dump,
                "batch norm": self.batch_norm.dump}

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features
