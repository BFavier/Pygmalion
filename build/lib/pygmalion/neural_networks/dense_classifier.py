import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from typing import List, Union
from .layers import BatchNorm1d, Linear, FullyConnected
from .conversions import dataframe_to_tensor, categories_to_tensor, \
                         floats_to_tensor, tensor_to_categories
from .nn_decorators import nn_classifier


@nn_classifier
class DenseClassifier(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        pass

    def __init__(self, inputs: List[str], categories: List[str],
                 hidden_layers: List[int] = [10, 10, 10],
                 activation: str = "relu"):
        super().__init__()
        self.inputs = list(inputs)
        self.input_norm = BatchNorm1d(len(inputs))
        self.hidden_layers = FullyConnected(len(inputs),
                                            hidden_layers=hidden_layers,
                                            activation=activation)
        self.categories = list(categories)
        self.output = Linear(self.hidden_layers.out_features,
                             len(self.categories))

    def forward(self, x):
        x = self.input_norm(x)
        x = self.hidden_layers(x)
        return self.output(x)

    def data_to_tensor(self, X: pd.DataFrame,
                       Y: Union[None, List[str]],
                       weights: Union[None, List[float]] = None
                       ) -> tuple:
        x = dataframe_to_tensor(X, self.inputs, self.device)
        y = None if Y is None else categories_to_tensor(Y, self.categories,
                                                        self.device)
        w = None if weights is None else floats_to_tensor(weights, self.device)
        return x, y, w

    def tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_categories(tensor, self.categories)

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor,
             weights: Union[None, torch.Tensor]) -> torch.Tensor:
        if weights is None:
            return F.cross_entropy(y_pred, y_target, weight=self.class_weights)
        else:
            return F.nll_loss(F.log_softmax(y_pred) * weights, y_target,
                              weight=self.class_weights)

    @property
    def dump(self):
        pass
