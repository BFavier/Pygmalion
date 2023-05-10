import torch
import pandas as pd
from typing import List, Union, Iterable, Optional
from ._conversions import classes_to_tensor, tensor_to_classes
from ._conversions import named_to_tensor, tensor_to_probabilities
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy
from .layers import Activation, Normalizer, FeaturesNorm, Dropout


class DenseClassifier(NeuralNetworkClassifier):

    def __init__(self, inputs: Iterable[str],
                 target: str, classes: Iterable[str],
                 hidden_layers: Iterable[int],
                 activation: str = "relu",
                 normalize: bool = True,
                 dropout: Optional[float] = None):
        """
        Parameters
        ----------
        inputs : Iterable of str
            the column names of the input variables in a dataframe
        target : str
            the ...
        ...

        """
        super().__init__(classes)
        self.inputs = tuple(inputs)
        self.target = str(target)
        self.layers = torch.nn.ModuleList()
        in_features = len(inputs)
        self.input_normalizer = Normalizer(1, in_features, affine=False)
        for out_features in hidden_layers:
            self.layers.append(torch.nn.Linear(in_features, out_features))
            if normalize:
                self.layers.append(FeaturesNorm(1, out_features))
            self.layers.append(Activation(activation))
            if dropout is not None:
                self.layers.append(Dropout(dropout))
            in_features = out_features
        out_features = len(self.classes)
        self.output = torch.nn.Linear(in_features, out_features)

    def forward(self, X: torch.Tensor):
        X = X.to(self.device)
        X = self.input_normalizer(X)
        for layer in self.layers:
            X = layer(X)
        return self.output(X)

    def loss(self, x: torch.Tensor, y_target: torch.Tensor,
             weights: Optional[torch.Tensor] = None):
        y_pred = self(x)
        return cross_entropy(y_pred, y_target, weights)

    @property
    def device(self) -> torch.device:
        return self.output.weight.device

    def _x_to_tensor(self, x: Union[pd.DataFrame, dict, Iterable],
                     device: Optional[torch.device] = None):
        return named_to_tensor(x, list(self.inputs), device=device)

    def _y_to_tensor(self, y: Union[pd.DataFrame, dict, Iterable[str]],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        if isinstance(y, dict):
            k, v = self.target, y[self.target]
            y = {k: v if hasattr(v, "__iter__") else [v]}
            y = pd.DataFrame.from_dict(y)
        if isinstance(y, pd.DataFrame):
            y = y[self.target]
        return classes_to_tensor(y, self.classes, device=device)

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_classes(tensor, self.classes)

    def _tensor_to_proba(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_probabilities(tensor, self.classes)
