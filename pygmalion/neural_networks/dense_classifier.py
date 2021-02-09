import torch
import pandas as pd
import numpy as np
from typing import List, Union
from .layers import BatchNorm1d, Linear, Dense0d
from .conversions import dataframe_to_tensor, classes_to_tensor, \
                         floats_to_tensor, tensor_to_classes
from .neural_network_classifier import NeuralNetworkClassifier
from .loss_functions import cross_entropy


class DenseClassifierModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.inputs = dump["inputs"]
        obj.classes = dump["classes"]
        obj.input_norm = BatchNorm1d.from_dump(dump["input norm"])
        obj.dense = Dense0d.from_dump(dump["dense"])
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self, inputs: List[str], classes: List[str],
                 hidden_layers: List[dict],
                 activation: str = "relu", stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        inputs : list of str
            The name of the dataframe columns used as inputs
        classes : list of str
            The unique classes the model can predict
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
        self.classes = list(classes)
        self.input_norm = BatchNorm1d(in_channels)
        self.dense = Dense0d(in_channels,
                             layers=hidden_layers,
                             activation=activation,
                             stacked=stacked,
                             dropout=dropout)
        in_channels = self.dense.out_channels(in_channels)
        self.output = Linear(in_channels,
                             len(self.classes))

    def forward(self, x):
        x = self.input_norm(x)
        x = self.dense(x)
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
        return cross_entropy(y_pred, y_target, weights=weights,
                             class_weights=self.class_weights)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "inputs": self.inputs,
                "classes": self.classes,
                "input norm": self.input_norm.dump,
                "dense": self.dense.dump,
                "output": self.output.dump}


class DenseClassifier(NeuralNetworkClassifier):

    ModuleType = DenseClassifierModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
