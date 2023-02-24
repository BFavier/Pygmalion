import torch
import pandas as pd
import numpy as np
from typing import Union, List, Iterable, Tuple, Optional
from .layers import ConvBlock, Upsampling2d
from ._conversions import tensor_to_index
from ._conversions import longs_to_tensor, images_to_tensor
from ._conversions import tensor_to_floats
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy


class ImageSegmenter(NeuralNetworkClassifier):

    def __init__(self, in_channels: int,
                 classes: Iterable[str],
                 features: Iterable[int],
                 kernel_size: Tuple[int, int] = (3, 3),
                 pooling_size: Optional[Tuple[int, int]] = (2, 2),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
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
        scale_factor = tuple(a*b for a, b in zip(stride, pooling_size or (1, 1)))
        in_features = in_channels
        for out_features in features:
            layer = torch.nn.ModuleDict(
                {"convolutions": ConvBlock(in_features, out_features, kernel_size, stride, activation,
                                           batch_norm, residuals, n_convs_per_block, dropout),
                 "downsampling": torch.nn.MaxPool2d(pooling_size) if pooling_size is not None else None})
            self.encoder.append(layer)
            in_features = out_features
        self.decoder = torch.nn.ModuleList()
        for out_features, add_features in zip(features[::-1], features[-2::-1]+[in_channels]):
            convolutions = ConvBlock(in_features+add_features, out_features,
                                     kernel_size, (1, 1), activation, batch_norm,
                                     residuals, n_convs_per_block, dropout)
            layer = torch.nn.ModuleDict(
                {"upsampling": Upsampling2d(scale_factor) if scale_factor != (1, 1) else None,
                 "convolutions": convolutions})
            self.decoder.append(layer)
            in_features = out_features
        self.output = torch.nn.Conv2d(out_features, len(self.classes), (1, 1))

    def forward(self, X: torch.Tensor):
        X = X.to(self.device)
        encoded = []
        for layer in self.encoder:
            encoded.append(X)
            convolutions, downsampling = layer["convolutions"], layer["downsampling"], 
            X = convolutions(X)
            if downsampling is not None:
                X = downsampling(X)
        for layer, feature_map in zip(self.decoder, encoded[::-1]):
            upsampling, convolutions = layer["upsampling"], layer["convolutions"]
            upsampled = upsampling(X) if upsampling is not None else X
            X = torch.cat([feature_map, upsampled], dim=1)
            X = convolutions(X)
        return self.output(X)

    def loss(self, x: torch.Tensor, y_target: torch.Tensor,
             weights: Optional[torch.Tensor] = None,
             class_weights: Optional[torch.Tensor] = None):
        y_pred = self(x)
        return cross_entropy(y_pred, y_target, weights, class_weights)
    
    @property
    def device(self) -> torch.device:
        return self.output.weight.device

    def _x_to_tensor(self, x: np.ndarray,
                     device: Optional[torch.device] = None):
        return images_to_tensor(x, device=device)

    def _y_to_tensor(self, y: Iterable[str],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        return longs_to_tensor(y, device=device)

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_index(tensor, dim=1)

    def _tensor_to_proba(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_floats(torch.softmax(tensor, dim=1), self.classes)