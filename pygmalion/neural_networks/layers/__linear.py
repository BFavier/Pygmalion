import torch


class Linear(torch.nn.Linear):
    """A wrapper around torch.nn.Linear"""

    @classmethod
    def from_dump(cls, dump: dict) -> 'Linear':
        """returns a 'Linear' layer from a dump"""
        assert dump["type"] == cls.__name__
        obj = cls(dump["in features"],
                  dump["out features"],
                  bias=(dump["bias"] is not None))
        obj.weight = torch.nn.Parameter(torch.tensor(dump["weight"],
                                                     dtype=torch.float))
        if obj.bias is not None:
            obj.bias = torch.nn.Parameter(torch.tensor(dump["bias"],
                                                       dtype=torch.float))
        return obj

    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        Parameters
        ----------
        in_features : int
            The number of input features
        out_features : int
            The number of output features
        bias : bool
            If False, there is not bias term in the linear equation:
            y = weight*x + bias
        """
        super().__init__(in_features, out_features, bias=bias)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "in features": self.in_features,
                "out features": self.out_features,
                "weight": self.weight.tolist(),
                "bias": self.bias if self.bias is None else self.bias.tolist()}
