import torch
import torch.nn.functional as F


class Activation(torch.nn.Module):
    """A module applying an activation function to its input"""

    @classmethod
    def from_dump(cls, dump: dict) -> 'Activation':
        """returns an 'Activation' object from a dump"""
        assert dump["type"] == cls.__name__
        return cls(dump["function name"])

    def __init__(self, name: str = "relu"):
        """
        Parameters
        ----------
        name : str {"relu", "leaky_relu", "elu", "tanh", "sigmoid"}
            The name of the activation function to apply
        """
        super().__init__()
        self.function_name = name
        if name == "identity":
            self.function = lambda x: x
        elif hasattr(torch, name):
            self.function = getattr(torch, name)
        else:
            self.function = getattr(F, name)

    def forward(self, input: torch.Tensor):
        """Sequence of operations applied by the Module"""
        return self.function(input)

    @property
    def dump(self) -> dict:
        """returns a dump of the layer"""
        return {"type": type(self).__name__,
                "function name": self.function_name}
