import torch
import pandas as pd
import numpy as np
from typing import List, Iterable, Tuple, Optional
from .layers.convolutions import ConvolutionalEncoder
from ._conversions import tensor_to_classes
from ._conversions import classes_to_tensor, images_to_tensor
from ._conversions import tensor_to_probabilities
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy


class ImageClassifier(NeuralNetworkClassifier):

    def __init__(self, in_channels: int,
                 classes: Iterable[str],
                 features: Iterable[int],
                 kernel_size: Tuple[int, int] = (3, 3),
                 pooling_size: Tuple[int, int] = (2, 2),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
                 n_convs_per_block: int = 1,
                 normalize: bool = True,
                 residuals: bool = True,
                 dropout: Optional[float] = None,
                 gradient_checkpointing: bool = True):
        """
        Parameters
        ----------
        ...
        """
        super().__init__(classes)
        self.encoder = ConvolutionalEncoder(
            in_channels, features, kernel_size, pooling_size, stride, activation,
            n_convs_per_block, normalize, residuals, dropout, gradient_checkpointing)
        self.output = torch.nn.Linear(features[-1], len(self.classes))

    def forward(self, X: torch.Tensor):
        X = X.to(self.device)
        X = self.encoder(X)
        N, C, H, W = X.shape
        X = X.reshape(N, C, -1).mean(dim=-1)
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
        return classes_to_tensor(y, self.classes, device=device)

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_classes(tensor, self.classes)

    def _tensor_to_proba(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_probabilities(tensor, self.classes)
