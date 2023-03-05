import torch
import pandas as pd
import numpy as np
from typing import Sequence, Union, Iterable, Optional
from ._conversions import floats_to_tensor, tensor_to_floats
from ._conversions import named_to_tensor, tensor_to_dataframe
from ._neural_network import NeuralNetwork
from ._loss_functions import MSE
from .layers import Activation, Normalizer


class DenseRegressor(NeuralNetwork):
    """
    DenseRegressor is an implementation of a fully connected NeuralNetwork
    for tabular regression.

    The data type of input 'x' and target 'y' are both a pd.DataFrame, 
    or a dictionary, or a np.ndarray of shape (n_observations, n_classes)

    The prediction output is a np.ndarray of shape (n_observations,)
    if there is a single target, and a pd.DataFrame if there are several
    """

    def __init__(self, inputs: Iterable[str],
                 target: Union[str, Iterable[str]],
                 hidden_layers: Iterable[int],
                 activation: str = "relu",
                 normalize: bool = True,
                 dropout: Optional[float] = None):
        """
        Parameters
        ----------
        inputs : Iterable of str
            the column names of the input variables in a dataframe
        target : str or Iterable of str
            the column name(s) of the variable(s) to predict
        activation : str or Callable or torch.nn.Module
            the activation function
        normalize : bool
            whether or not to add normalization layers
        dropout : float or None
            the dropout after each hidden layer if provided
        """
        super().__init__()
        self.inputs = tuple(inputs)
        self.target = target if isinstance(target, str) else tuple(target)
        self.layers = torch.nn.ModuleList()
        in_features = len(inputs)
        if normalize:
            self.layers.append(Normalizer(in_features, affine=False))
        for out_features in hidden_layers:
            self.layers.append(torch.nn.Linear(in_features, out_features))
            if normalize:
                self.layers.append(Normalizer(out_features))
            self.layers.append(Activation(activation))
            if dropout is not None:
                self.layers.append(torch.nn.Dropout(dropout))
            in_features = out_features
        out_features = 1 if isinstance(target, str) else len(self.target)
        self.output = torch.nn.Linear(in_features, out_features)
        if normalize:
            self.target_norm = Normalizer(out_features, affine=False)
        else:
            self.target_norm = None

    def forward(self, X: torch.Tensor):
        X = X.to(self.device)
        for layer in self.layers:
            X = layer(X)
        return self.output(X)

    def loss(self, x: torch.Tensor, y_target: torch.Tensor,
             weights: Optional[torch.Tensor] = None):
        y_pred = self(x)
        if self.target_norm is not None:
            y_target = self.target_norm(y_target)
        return MSE(y_pred, y_target, weights)

    @property
    def device(self) -> torch.device:
        return self.output.weight.device

    def _x_to_tensor(self, x: Union[pd.DataFrame, dict, Iterable],
                     device: Optional[torch.device] = None):
        return named_to_tensor(x, list(self.inputs), device=device)

    def _y_to_tensor(self, y: Union[pd.DataFrame, dict, Iterable],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        target = [self.target] if isinstance(self.target, str) else list(self.target)
        return named_to_tensor(y, target, device=device)

    def _tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        if self.target_norm is not None:
            tensor = self.target_norm.unscale(tensor)
        if isinstance(self.target, str):
            return tensor_to_floats(tensor).reshape(-1)
        else:
            return tensor_to_dataframe(tensor, self.target)
