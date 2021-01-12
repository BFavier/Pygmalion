import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple
from .layers import Linear, BatchNorm2d
from .layers import Encoder2d, OverallPool2d, FullyConnected
from .conversions import floats_to_tensor, tensor_to_index
from .conversions import categories_to_tensor, images_to_tensor
from .nn_decorators import nn_classifier


@nn_classifier
class ImageClassifier(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        pass

    def __init__(self, in_channels: int,
                 categories: List[str],
                 convolutions: Union[List[dict], List[List[dict]]],
                 pooling: List[Tuple[int, int]],
                 fully_connected: List[int] = [],
                 pooling_type: str = "Max",
                 padded: bool = True,
                 activation: str = "relu"):
        super().__init__()
        assert len(convolutions) == len(pooling)
        self.in_channels = in_channels
        self.categories = list(categories)
        self.input_norm = BatchNorm2d(in_channels)
        self.encoder = Encoder2d(in_channels, convolutions, pooling,
                                 pooling_type, padded=padded,
                                 activation=activation)
        self.overall_pool = OverallPool2d(pooling_type)
        self.fully_connected = FullyConnected(self.encoder.out_channels,
                                              fully_connected,
                                              activation=activation)
        self.output = Linear(self.fully_connected.out_features,
                             len(categories))

    def forward(self, X: torch.Tensor):
        X = self.input_norm(X)
        X = self.encoder(X)
        X = self.overall_pool(X)
        X = self.fully_connected(X)
        return self.output(X)

    def data_to_tensor(self, X: np.ndarray,
                       Y: Union[None, List[str]],
                       weights: Union[None, List[float]] = None
                       ) -> tuple:
        x = images_to_tensor(X, self.device)
        y = None if Y is None else categories_to_tensor(Y, self.categories,
                                                        self.device)
        w = None if weights is None else floats_to_tensor(weights)
        return x, y, w

    def tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        indexes = tensor_to_index(tensor)
        return [self.categories[i] for i in indexes]

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor,
             weights: Union[torch.Tensor, None]):
        if weights is None:
            return F.cross_entropy(y_pred, y_target, weight=self.class_weights)
        else:
            return F.nll_loss(F.log_softmax(y_pred, dim=-1) * weights,
                              y_target, weight=self.class_weights)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "shape in": self.shape_in,
                "categories": list(self.categories),
                "input norm": self.input_norm.dump,
                "convolutional": [c.dump for c in self.convolutional],
                "fully connected": [d.dump for d in self.fully_connected],
                "output": self.output.dump}
