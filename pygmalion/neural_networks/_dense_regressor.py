import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Sequence, Union, Iterable, Optional
from ._conversions import floats_to_tensor, tensor_to_floats
from ._conversions import named_to_tensor, tensor_to_dataframe
from ._neural_network import NeuralNetwork
from ._loss_functions import RMSE
from .layers import Activation


class DenseRegressor(NeuralNetwork):

    def __init__(self, input_names: Iterable[str],
                 target_names: Union[str, Iterable[str]],
                 hidden_layers: Iterable[int],
                 activation: str = "relu",
                 batch_norm: bool = True,
                 dropout: Optional[float] = None):
        super().__init__()
        self.input_names = tuple(input_names)
        self.target_names = target_names
        self.layers = torch.nn.ModuleList()
        in_features = len(input_names)
        if batch_norm:
            self.layers.append(torch.nn.BatchNorm1d(in_features))
        for out_features in hidden_layers:
            self.layers.append(torch.nn.Linear(in_features, out_features))
            if batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(out_features))
            self.layers.append(Activation(activation))
            if dropout is not None:
                self.layers.append(torch.nn.Dropout(dropout))
            in_features = out_features
        out_features = 1 if isinstance(target_names, str) else len(self.target_names)
        self.output = torch.nn.Linear(in_features, out_features)
        if batch_norm:
            self.target_norm = torch.nn.BatchNorm1d(out_features, affine=False)
        else:
            self.target_norm = None

    def forward(self, X: torch.Tensor):
        for layer in self.layers:
            X = layer(X)
        return self.output(X)

    def loss(self, x: torch.Tensor, y_target: torch.Tensor):
        y_pred = self(x)
        if self.target_norm is not None:
            y_target = self.target_norm(y_target)
        return F.mse_loss(y_pred, y_target)

    def data_to_tensor(self, df: Union[pd.DataFrame, dict],
                        weights: Optional[Sequence[float]] = None,
                        device: Optional[torch.device] = None) -> tuple:
        x = self._x_to_tensor(df, device=device)
        y = self._y_to_tensor(df, device=device)
        if weights is not None:
            w = floats_to_tensor(weights, device)
            data = (x, y, w/w.mean())
        else:
            data = (x, y)
        return data

    def _x_to_tensor(self, x: Union[pd.DataFrame, dict, Iterable],
                     device: Optional[torch.device] = None):
        return named_to_tensor(x, list(self.input_names), device=device)

    def _y_to_tensor(self, y: Union[pd.DataFrame, dict, Iterable],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        if isinstance(self.target_names, str):
            return floats_to_tensor(y[self.target_names], device=device).unsqueeze(-1)
        else:
            return named_to_tensor(y, list(self.target_names), device=device)

    def _tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        if self.target_norm is not None:
            tensor = (tensor * (self.target_norm.running_var + self.target_norm.eps)**0.5
                      + self.target_norm.running_mean)
        if isinstance(self.target_names, str):
            return tensor_to_floats(tensor).reshape(-1)
        else:
            return tensor_to_dataframe(tensor, self.target_names)
