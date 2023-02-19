import torch
import numpy as np
import pandas as pd
from typing import Union, List, Iterable
from .layers import ConvBlock
from ._conversions import floats_to_tensor, tensor_to_index
from ._conversions import classes_to_tensor, images_to_tensor
from ._conversions import tensor_to_probabilities
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy
from pygmalion.utilities import document


class ImageClassifier(NeuralNetworkClassifier):

    def __init__(self, in_channels: int,
                 classes: List[str],
                 layers: Iterable[torch.nn.Module]):
        """
        Parameters
        ----------
        ...
        """
        super().__init__()
        assert len(pooling) == len(convolutions) - 1
        self.classes = tuple(classes)
        self.blocks = torch.nn.ModuleList()
        pass

    def forward(self, X: torch.Tensor):
        X = self.input_norm(X)
        X = self.encoder(X)
        X = self.dense(X)
        return self.output(X)

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor,
             weights: Union[None, torch.Tensor] = None):
        return cross_entropy(y_pred, y_target, weights,
                             self.class_weights)
