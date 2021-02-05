import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple
from .layers import Linear, BatchNorm2d
from .layers import Encoder2d, OverallPool2d, FullyConnected
from .conversions import floats_to_tensor, tensor_to_index
from .conversions import classes_to_tensor, images_to_tensor
from .nn_decorators import nn_classifier


@nn_classifier
class ImageClassifier(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls(1, dump["classes"], [], [])
        obj.input_norm = BatchNorm2d.from_dump(dump["input norm"])
        obj.encoder = Encoder2d.from_dump(dump["encoder"])
        obj.overall_pool = OverallPool2d.from_dump(dump["overall pool"])
        obj.fully_connected = FullyConnected.from_dump(dump["fully connected"])
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self, in_channels: int,
                 classes: List[str],
                 convolutions: Union[List[dict], List[List[dict]]],
                 pooling: List[Tuple[int, int]],
                 fully_connected: List[int] = [],
                 pooling_type: str = "Max",
                 padded: bool = True,
                 activation: str = "relu"):
        super().__init__()
        assert len(convolutions) == len(pooling)
        self.classes = list(classes)
        self.input_norm = BatchNorm2d(in_channels)
        self.encoder = Encoder2d(in_channels, convolutions, pooling,
                                 pooling_type, padded=padded,
                                 activation=activation)
        self.overall_pool = OverallPool2d(pooling_type)
        self.fully_connected = FullyConnected(self.encoder.out_channels,
                                              fully_connected,
                                              activation=activation)
        self.output = Linear(self.fully_connected.out_features,
                             len(classes))

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
        y = None if Y is None else classes_to_tensor(Y, self.classes,
                                                        self.device)
        w = None if weights is None else floats_to_tensor(weights, self.device)
        return x, y, w

    def tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        indexes = tensor_to_index(tensor)
        return [self.classes[i] for i in indexes]

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
                "classes": list(self.classes),
                "input norm": self.input_norm.dump,
                "encoder": self.encoder.dump,
                "overall pool": self.overall_pool.dump,
                "fully connected": self.fully_connected.dump,
                "output": self.output.dump}
