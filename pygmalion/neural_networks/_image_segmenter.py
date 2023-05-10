import torch
import pandas as pd
import numpy as np
from typing import List, Iterable, Tuple, Optional
from .layers.convolutions import ConvBlock, Upsampling2d, PaddedConv2d, UPSAMPLING_METHOD
from .layers.convolutions import ConvolutionalEncoder, ConvolutionalDecoder
from ._conversions import tensor_to_index
from ._conversions import longs_to_tensor, images_to_tensor
from ._conversions import tensor_to_floats
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy, soft_dice_loss


class ImageSegmenter(NeuralNetworkClassifier):

    def __init__(self, in_channels: int,
                 classes: Iterable[str],
                 features: Iterable[int],
                 kernel_size: Tuple[int, int] = (3, 3),
                 pooling_size: Optional[Tuple[int, int]] = (2, 2),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
                 n_convs_per_block: int = 1,
                 normalize: bool = True,
                 residuals: bool = True,
                 upsampling_method: UPSAMPLING_METHOD = "nearest",
                 dropout: Optional[float] = None,
                 entropy_dice_mixture: float = 0.9,
                 low_memory: bool = True):
        """
        Parameters
        ----------
        ...
        """
        super().__init__(classes)
        self.entropy_dice_mixture = entropy_dice_mixture
        scale_factor = tuple(a*b for a, b in zip(stride, pooling_size or (1, 1)))
        self.encoder = ConvolutionalEncoder(in_channels, features, kernel_size,
                                            pooling_size, stride, activation,
                                            n_convs_per_block, normalize,
                                            residuals, dropout, low_memory)
        self.intermediate = ConvBlock(
            features[-1], features[-1], kernel_size, stride, activation,
            normalize, residuals, n_convs_per_block, dropout)
        self.decoder = ConvolutionalDecoder(features[-1], features[::-1], features[::-1],
                                            kernel_size, scale_factor, upsampling_method,
                                            activation, n_convs_per_block, normalize,
                                            residuals, dropout, low_memory)
        self.output = torch.nn.Conv2d(features[0], len(self.classes), (1, 1))

    def forward(self, X: torch.Tensor):
        X = X.to(self.device)
        encoded = []
        X = self.encoder(X, encoded)
        X = self.intermediate(X)
        X = self.decoder(X, encoded[::-1])
        return self.output(X)

    def loss(self, x: torch.Tensor, y_target: torch.Tensor,
             weights: Optional[torch.Tensor] = None,
             class_weights: Optional[torch.Tensor] = None):
        assert 0.0 <= self.entropy_dice_mixture <= 1.0
        assert ((0 <= y_target) & (y_target < len(self.classes))).all()
        alpha = self.entropy_dice_mixture
        y_pred = self(x)
        return alpha * cross_entropy(y_pred, y_target, weights, class_weights) + (1-alpha) * soft_dice_loss(y_pred, y_target, weights, class_weights)

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