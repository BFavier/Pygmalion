import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from typing import List, Union
from .layers import BatchNorm1d, Linear, FullyConnected
from .conversions import dataframe_to_tensor, classes_to_tensor, \
                         floats_to_tensor, tensor_to_classes
from .nn_decorators import nn_classifier


@nn_classifier
class DenseClassifier(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls(dump["inputs"], dump["classes"])
        obj.input_norm = BatchNorm1d.from_dump(dump["input norm"])
        obj.fully_connected = FullyConnected.from_dump(dump["fully connected"])
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self, inputs: List[str], classes: List[str],
                 hidden_layers: List[int] = [10, 10, 10],
                 activation: str = "relu"):
        super().__init__()
        self.inputs = list(inputs)
        self.classes = list(classes)
        self.input_norm = BatchNorm1d(len(inputs))
        self.fully_connected = FullyConnected(len(inputs),
                                              hidden_layers=hidden_layers,
                                              activation=activation)
        self.output = Linear(self.fully_connected.out_features,
                             len(self.classes))

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fully_connected(x)
        return self.output(x)

    def data_to_tensor(self, X: pd.DataFrame,
                       Y: Union[None, List[str]],
                       weights: Union[None, List[float]] = None
                       ) -> tuple:
        x = dataframe_to_tensor(X, self.inputs, self.device)
        y = None if Y is None else classes_to_tensor(Y, self.classes,
                                                        self.device)
        w = None if weights is None else floats_to_tensor(weights, self.device)
        return x, y, w

    def tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_classes(tensor, self.classes)

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor,
             weights: Union[None, torch.Tensor]) -> torch.Tensor:
        if weights is None:
            return F.cross_entropy(y_pred, y_target, weight=self.class_weights)
        else:
            return F.nll_loss(F.log_softmax(y_pred) * weights, y_target,
                              weight=self.class_weights)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "inputs": self.inputs,
                "classes": self.classes,
                "input norm": self.input_norm.dump,
                "fully connected": self.fully_connected.dump,
                "output": self.output.dump}
