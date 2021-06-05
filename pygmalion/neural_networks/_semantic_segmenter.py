import torch as _torch
import numpy as _np
from typing import Union, List, Tuple, Dict, Iterable
from .layers import BatchNorm2d, Conv2d
from .layers import UNet2d
from ._conversions import floats_to_tensor, tensor_to_index
from ._conversions import segmented_to_tensor, images_to_tensor
from ._neural_network_classifier import NeuralNetworkClassifier
from ._loss_functions import soft_dice_loss
from pygmalion.utilities import document


class SemanticSegmenterModule(_torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        _torch.nn.Module.__init__(obj)
        obj.colors = dump["colors"]
        obj.classes = dump["classes"]
        obj.input_norm = BatchNorm2d.from_dump(dump["input norm"])
        obj.u_net = UNet2d.from_dump(dump["u-net"])
        obj.output = Conv2d.from_dump(dump["output"])
        return obj

    def __init__(self, in_features: int,
                 colors: Dict[str, Union[int, List[int]]],
                 downsampling: List[Union[dict, List[dict]]],
                 pooling: List[Tuple[int, int]],
                 upsampling: List[Union[dict, List[dict]]],
                 pooling_type: str = "max",
                 upsampling_method: str = "nearest",
                 activation: str = "relu",
                 stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        in_features : int
            The number of channels of the input
        colors : dict
            a dict of {class: color}
        downsampling : list of [dict / list of dict]
            the kwargs for the 'Activated2d' layers for all 'downsampling'
        pooling : list of [int / tuple of int]
            the pooling window of all downsampling layers
        upsampling : list of [dict / list of dict]
            the kwargs for the 'Activated2d' layers for all 'upsampling'
        pooling_type : one of {'max', 'avg'}
            the type of pooling
        upsampling_method : one of {'nearest', 'interpolate'}
            the method used for the unpooling layers
        activation : str
            the default value for the 'activation' key of the kwargs
        stacked : bool
            the default value for the 'stacked' key of the kwargs
        dropout : float or None
            the default value for the 'dropout' key of the kwargs
        """
        super().__init__()
        self.classes = [c for c in colors.keys()]
        self.colors = [colors[c] for c in self.classes]
        self.input_norm = BatchNorm2d(in_features)
        self.u_net = UNet2d(in_features, downsampling, pooling, upsampling,
                            pooling_type=pooling_type,
                            upsampling_method=upsampling_method,
                            activation=activation,
                            stacked=stacked,
                            dropout=dropout)
        in_features = self.u_net.out_features(in_features)
        self.output = Conv2d(in_features, len(self.classes), (1, 1))

    def forward(self, X: _torch.Tensor):
        X = self.input_norm(X)
        X = self.u_net(X)
        return self.output(X)

    def loss(self, y_pred: _torch.Tensor, y_target: _torch.Tensor,
             weights: Union[None, _torch.Tensor] = None):
        return soft_dice_loss(y_pred, y_target, weights,
                              self.class_weights)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "classes": list(self.classes),
                "colors": list(self.colors),
                "input norm": self.input_norm.dump,
                "u-net": self.u_net.dump,
                "output": self.output.dump}


class SemanticSegmenter(NeuralNetworkClassifier):

    ModuleType = SemanticSegmenterModule

    @document(ModuleType.__init__, NeuralNetworkClassifier.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _data_to_tensor(self, X: Iterable[_np.ndarray],
                        Y: Union[None, List[str]],
                        weights: Union[None, List[float]] = None,
                        device: _torch.device = _torch.device("cpu")) -> tuple:
        x = images_to_tensor(X, device)
        y = None if Y is None else segmented_to_tensor(Y, self.module.colors,
                                                       device)
        w = None if weights is None else floats_to_tensor(weights, device)
        return x, y, w

    def _tensor_to_y(self, tensor: _torch.Tensor) -> _np.ndarray:
        indexes = tensor_to_index(tensor)
        return _np.array(self.module.colors)[indexes]
