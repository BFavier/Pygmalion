import torch
import pandas as pd
import numpy as np
from typing import Union, List, Iterable, Tuple, Optional
from .layers import ConvBlock, Upsampling2d
from ._conversions import tensor_to_classes
from ._conversions import classes_to_tensor, images_to_tensor
from ._conversions import tensor_to_probabilities
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy


class ImageSegmenter(NeuralNetworkClassifier):

    def __init__(self, in_channels: int,
                 classes: Iterable[str],
                 features: Iterable[int],
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
                 pooling_size: Optional[Tuple[int, int]] = None,
                 n_convs_per_block: int = 1,
                 batch_norm: bool = True,
                 residuals: bool = True,
                 dropout: Optional[float] = None):
        """
        Parameters
        ----------
        ...
        """
        super().__init__()
        self.classes = tuple(classes)
        self.encoder = torch.nn.ModuleList()
        scale_factor = (a*b for a, b in zip(stride, pooling_size or (1, 1)))
        in_features = in_channels
        for out_features in features:
            self.encoder.append(
                ConvBlock(in_features, out_features, kernel_size, stride, activation,
                          batch_norm, residuals, n_convs_per_block, dropout))
            if pooling_size is not None:
                self.encoder.append(torch.nn.MaxPool2d(pooling_size))
            in_features = out_features
        self.decoder = torch.nn.ModuleList()
        for out_features, add_features in zip(features[::-1], features[-2::-1]+[in_channels]):
            convolutions = ConvBlock(in_features+add_features, out_features,
                                     kernel_size, stride, activation, batch_norm,
                                     residuals, n_convs_per_block, dropout)
            layer = torch.nn.ModuleDict({"upsampling": Upsampling2d(scale_factor),
                                         "convolutions": convolutions})
            self.decoder.append(layer)
            in_features = out_features
        self.output = torch.nn.Conv2d(out_features, len(self.classes), (1, 1))

    def forward(self, X: torch.Tensor):
        encoded = []
        for layer in self.encoder:
            encoded.append(X)
            X = layer(X)
        for layer, feature_map in zip(self.decoder, encoded[::-1]):
            X = torch.cat([feature_map, layer["upsampling"](X)], dim=1)
            X = layer["convolutions"](X)
        return self.output(X)

    def loss(self, x: torch.Tensor, y_target: torch.Tensor,
             weights: Optional[torch.Tensor] = None,
             class_weights: Optional[torch.Tensor] = None):
        y_pred = self(x)
        return cross_entropy(y_pred, y_target, weights, class_weights)

    def _x_to_tensor(self, x: np.ndarray,
                     device: Optional[torch.device] = None):
        return images_to_tensor(x, device=device)

    def _y_to_tensor(self, y: Iterable[str],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        return classes_to_tensor(y, self.classes, device=device)

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_classes(tensor, self.classes)

    def _tensor_to_proba(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_probabilities(tensor, self.classes)