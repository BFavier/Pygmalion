import torch
import numpy as np
from typing import Union, List, Tuple
from .layers import Conv2d, BatchNorm2d
from .layers import Encoder2d, Dense2d
from .conversions import floats_to_tensor, tensor_to_index
from .conversions import classes_to_tensor, images_to_tensor
from .neural_network_classifier import NeuralNetworkClassifier
from .loss_functions import cross_entropy


class ObjectDetectorModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.classes = dump["classes"]
        obj.input_norm = BatchNorm2d.from_dump(dump["input norm"])
        obj.encoder = Encoder2d.from_dump(dump["encoder"])
        obj.dense = Dense2d.from_dump(dump["dense"])
        obj.output = Conv2d.from_dump(dump["output"])
        return obj

    def __init__(self, in_channels: int,
                 classes: List[str],
                 convolutions: Union[List[dict], List[List[dict]]],
                 pooling: List[Tuple[int, int]],
                 dense: List[dict],
                 pooling_type: str = "max",
                 padded: bool = True,
                 activation: str = "relu",
                 stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        in_channels : int
            the number of channels in the input images
        class : list of str
            the unique classes the model can predict
        convolutions : list of [dict / list of dict]
            the kwargs for the 'Activated2d' layers for all 'downsampling'
        pooling : list of [int / tuple of int]
            the pooling window of all downsampling layers
            (excepted the last one which is an overall pool)
        dense : list of dict
            the kwargs for the 'Activated2d' of the final 'Dense2d' layer
        pooling_type : one of {'max', 'avg'}
            the type of pooling
        padded : bool
            the default value for the 'padded' key of the kwargs
        activation : str
            the default value for the 'activation' key of the kwargs
        stacked : bool
            the default value for the 'stacked' key of the kwargs
        dropout : float or None
            the default value for the 'dropout' key of the kwargs
        """
        super().__init__()
        assert len(pooling) == len(convolutions) - 1
        self.classes = list(classes)
        self.input_norm = BatchNorm2d(in_channels)
        self.encoder = Encoder2d(in_channels, convolutions, pooling+[None],
                                 pooling_type=pooling_type,
                                 padded=padded,
                                 activation=activation,
                                 stacked=stacked,
                                 dropout=dropout)
        in_channels = self.encoder.out_channels(in_channels)
        self.dense = Dense2d(in_channels, dense, activation=activation,
                             stacked=stacked, dropout=dropout)
        in_channels = self.dense.out_channels(in_channels)
        self.output = Conv2d(in_channels, len(classes), kernel_size=(1, 1))

    def forward(self, X: torch.Tensor):
        X = self.input_norm(X)
        X = self.encoder(X)
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
        return cross_entropy(y_pred, y_target, weights=weights,
                             class_weights=self.class_weights)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "classes": list(self.classes),
                "input norm": self.input_norm.dump,
                "encoder": self.encoder.dump,
                "dense": self.dense.dump,
                "output": self.output.dump}


class ObjectDetector(NeuralNetworkClassifier):

    ModuleType = ObjectDetectorModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
