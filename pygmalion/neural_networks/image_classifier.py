import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple
from .layers import Linear, BatchNorm2d
from .layers import Encoder2d, Pooling2d, Dense0d
from .conversions import floats_to_tensor, tensor_to_index
from .conversions import classes_to_tensor, images_to_tensor
from .neural_network_classifier import NeuralNetworkClassifier


class ImageClassifierModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        obj.classes = dump["classes"]
        obj.input_norm = BatchNorm2d.from_dump(dump["input norm"])
        obj.encoder = Encoder2d.from_dump(dump["encoder"])
        obj.final_pool = Pooling2d.from_dump(dump["final pool"])
        obj.dense = Dense0d.from_dump(dump["dense"])
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self, in_channels: int,
                 classes: List[str],
                 convolutions: Union[List[dict], List[List[dict]]],
                 pooling: List[Tuple[int, int]],
                 dense: List[dict] = [],
                 pooling_type: str = "max",
                 padded: bool = True,
                 activation: str = "relu",
                 stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        in_channels : int
        """
        super().__init__()
        assert len(convolutions) == len(pooling)
        self.classes = list(classes)
        self.input_norm = BatchNorm2d(in_channels)
        self.encoder = Encoder2d(in_channels, convolutions, pooling,
                                 pooling_type=pooling_type,
                                 padded=padded,
                                 activation=activation,
                                 stacked=stacked,
                                 dropout=dropout)
        in_channels = self.encoder.out_channels(in_channels)
        self.final_pool = Pooling2d(None, pooling_type)
        self.dense = Dense0d(in_channels, dense, activation=activation)
        in_channels = self.dense.out_channels(in_channels)
        self.output = Linear(in_channels, len(classes))

    def forward(self, X: torch.Tensor):
        X = self.input_norm(X)
        X = self.encoder(X)
        X = self.final_pool(X)
        X = self.dense(X)
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
            return F.nll_loss(F.log_softmax(y_pred, dim=1) * weights,
                              y_target, weight=self.class_weights)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "classes": list(self.classes),
                "input norm": self.input_norm.dump,
                "encoder": self.encoder.dump,
                "final pool": self.final_pool.dump,
                "dense": self.dense.dump,
                "output": self.output.dump}


class ImageClassifier(NeuralNetworkClassifier):

    ModuleType = ImageClassifierModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
