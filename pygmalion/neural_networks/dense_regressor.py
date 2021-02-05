import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from typing import List, Union, Iterable
from .layers import Dense0d, BatchNorm0d, Linear
from .conversions import dataframe_to_tensor, \
                         floats_to_tensor, tensor_to_floats
from .neural_network import NeuralNetwork
# from .nn_decorators import neural_network


class DenseRegressorModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        obj.inputs = dump["inputs"]
        obj.input_norm = BatchNorm0d.from_dump(dump["input norm"])
        obj.dense = Dense0d.from_dump(dump["dense"])
        obj.output = Linear.from_dump(dump["output"])
        obj.target_norm = BatchNorm0d.from_dump(dump["target norm"])
        return obj

    def __init__(self, inputs: List[str],
                 hidden_layers: List[dict],
                 activation: str = "relu", stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        inputs : list of str
            The name of the dataframe columns used as inputs
        hidden_layers : list of dict
            The kwargs of all 'Activated0d' layers
        activation : str
            default 'activation' parameter for 'Activated0d' layers
        stacked : bool
            default 'stacked' parameter for 'Activated0d' layers
        dropout : None or float
            default 'dropout' parameter for 'Activated0d' layers
        """
        super().__init__()
        self.inputs = list(inputs)
        in_channels = len(inputs)
        self.input_norm = BatchNorm0d(in_channels)
        self.dense = Dense0d(in_channels,
                             layers=hidden_layers,
                             activation=activation,
                             stacked=stacked,
                             dropout=dropout)
        in_channels = self.dense.out_channels(in_channels)
        self.output = Linear(in_channels, 1)
        self.target_norm = BatchNorm0d(1)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.dense(x)
        return self.output(x)

    def data_to_tensor(self, X: Union[pd.DataFrame, Iterable],
                       Y: Union[None, np.ndarray],
                       weights: Union[None, List[float]] = None
                       ) -> tuple:
        if isinstance(X, pd.DataFrame):
            x = dataframe_to_tensor(X, self.inputs, self.device)
        else:
            x = floats_to_tensor(X, self.device)
        y = None if Y is None else floats_to_tensor(Y, self.device).view(-1, 1)
        if weights is None:
            w = None
        else:
            w = floats_to_tensor(weights, self.device).view(-1, 1)
        return x, y, w

    def tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_floats(self.target_norm.undo(tensor).view(-1))

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor,
             weights: Union[None, torch.Tensor]) -> torch.Tensor:
        y_target = self.target_norm(y_target)
        if weights is None:
            return torch.sqrt(F.mse_loss(y_pred, y_target))
        else:
            return torch.sqrt(torch.mean(weights * (y_pred - y_target)**2))

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "inputs": self.inputs,
                "input norm": self.input_norm.dump,
                "dense": self.dense.dump,
                "output": self.output.dump,
                "target norm": self.target_norm.dump}


class DenseRegressor(NeuralNetwork):

    ModuleType = DenseRegressorModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
