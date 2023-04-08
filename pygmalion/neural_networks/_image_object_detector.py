import torch
import pandas as pd
import numpy as np
from typing import List, Sequence, Iterable, Tuple, Optional
from .layers import ConvBlock
from ._conversions import tensor_to_classes
from ._conversions import classes_to_tensor, images_to_tensor
from ._conversions import tensor_to_probabilities
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy


class ImageObjectDetector(NeuralNetworkClassifier):

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
                 dropout: Optional[float] = None):
        """
        Parameters
        ----------
        ...
        """
        super().__init__(classes)
        self.downscaling_factor = tuple((s*mp) for s, mp in zip(stride, pooling_size or (1, 1)))
        self.n_stages = len(features)
        self.layers = torch.nn.ModuleList()
        in_features = in_channels
        for out_features in features:
            self.layers.append(
                ConvBlock(in_features, out_features, kernel_size, stride, activation,
                          normalize, residuals, n_convs_per_block, dropout))
            if pooling_size is not None:
                self.layers.append(torch.nn.MaxPool2d(pooling_size))
            in_features = out_features
        self.detected = torch.nn.Conv2d(out_features, 1, (1, 1))
        self.positions = torch.nn.Conv2d(out_features, 2, (1, 1))
        self.dimensions = torch.nn.Conv2d(out_features, 2, (1, 1))
        self.classes = torch.nn.Conv2d(out_features, len(self.classes), (1, 1))

    def forward(self, X: torch.Tensor):
        X = X.to(self.device)
        for layer in self.layers:
            X = layer(X)
        N, C, H, W = X.shape
        X = X.reshape(N, C, -1).mean(dim=-1)
        return self.detected(X), self.positions(X), self.dimensions(X), self.classes(X)

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

    def _y_to_tensor(self, x: np.ndarray, y: Iterable[dict],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        grid_h, grid_w = (f ** len(self.n_stages) for f in self.downscaling_factor)
        n, h, w = x.shape[:3]
        h, w = (h//grid_h, w//grid_w)
        detected = torch.zeros((n, h, w), dtype=torch.bool)
        positions = torch.zeros((n, h, w, 2), dtype=torch.float)
        dimensions = torch.zeros((n, h, w, 2), dtype=torch.float)
        classes = torch.zeros((n, h, w), dtype=torch.long)
        for n, bboxes in enumerate(y):
            X, Y, W, H = (torch.tensor(bboxes[v]) for v in ("x", "y", "w", "h"))
            C = classes_to_tensor(y, bboxes["class"], device=device)
            i, j = (Y // grid_h, X // grid_w)
            py, px = (Y % grid_h, X % grid_w)
            detected[n, i, j] = 1
            positions[n, i, j, :] = torch.stack([px, py], dim=1)
            dimensions[n, i, j, :] = torch.stack([W / grid_w, H / grid_h], dim=1)
            classes[n, i, j] = C
        return detected, positions, dimensions, classes

    def data_to_tensor(self, x: np.ndarray, y: List[dict],
                       weights: Optional[Sequence[float]] = None,
                       class_weights: Optional[Sequence[float]] = None,
                       device: Optional[torch.device] = None, **kwargs) -> tuple:
        images = self._x_to_tensor(x, device, **kwargs)
        detected, positions, dimensions, classes = self._y_to_tensor()
        return detected, positions, dimensions, classes

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_classes(tensor, self.classes)

    def _tensor_to_proba(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_probabilities(tensor, self.classes)
