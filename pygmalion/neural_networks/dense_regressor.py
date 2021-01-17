import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from typing import List, Union, Iterable
from .layers import FullyConnected, BatchNorm1d, Linear
from .conversions import dataframe_to_tensor, \
                         floats_to_tensor, tensor_to_floats
from .nn_decorators import neural_network


@neural_network
class DenseRegressor(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls(dump["inputs"])
        obj.input_norm = BatchNorm1d.from_dump(dump["input norm"])
        obj.hidden_layers = FullyConnected.from_dump(dump["fully connected"])
        obj.output = Linear.from_dump(dump["output"])
        obj.target_norm = BatchNorm1d.from_dump(dump["target norm"])
        return obj

    def __init__(self, inputs: List[str],
                 hidden_layers: List[int] = [10, 10, 10],
                 activation: str = "relu"):
        super().__init__()
        self.inputs = list(inputs)
        self.input_norm = BatchNorm1d(len(inputs))
        self.fully_connected = FullyConnected(len(inputs),
                                              hidden_layers=hidden_layers,
                                              activation=activation)
        self.output = Linear(self.fully_connected.out_features, 1)
        self.target_norm = BatchNorm1d(1)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fully_connected(x)
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
        w = None if weights is None else floats_to_tensor(weights, self.device).view(-1, 1)
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
                "hidden layers": self.hidden_layers.dump,
                "output": self.output.dump,
                "target norm": self.target_norm.dump}
