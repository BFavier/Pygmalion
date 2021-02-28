import torch
import numpy as np
from typing import Union, List, Tuple, Dict, Iterable
from .layers import BatchNorm2d, Conv2d
from .layers import UNet2d
from .conversions import floats_to_tensor, tensor_to_index
from .conversions import segmented_to_tensor, images_to_tensor
from .neural_network_classifier import NeuralNetworkClassifier
from .loss_functions import soft_dice_loss


class SemanticSegmenterModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.colors = dump["colors"]
        obj.classes = dump["classes"]
        obj.input_norm = BatchNorm2d.from_dump(dump["input norm"])
        obj.u_net = UNet2d.from_dump(dump["u-net"])
        obj.output = Conv2d.from_dump(dump["output"])
        return obj

    def __init__(self, in_channels: int,
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
        in_channels : int
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
        self.input_norm = BatchNorm2d(in_channels)
        self.u_net = UNet2d(in_channels, downsampling, pooling, upsampling,
                            pooling_type=pooling_type,
                            upsampling_method=upsampling_method,
                            activation=activation,
                            stacked=stacked,
                            dropout=dropout)
        in_channels = self.u_net.out_channels(in_channels)
        self.output = Conv2d(in_channels, len(self.classes), (1, 1))

    def forward(self, X: torch.Tensor):
        X = self.input_norm(X)
        X = self.u_net(X)
        return self.output(X)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _loss_function(self, y_pred: torch.Tensor, y_target: torch.Tensor,
                       weights: Union[None, torch.Tensor] = None):
        return soft_dice_loss(y_pred, y_target, weights,
                              self.module.class_weights)

    def _data_to_tensor(self, X: Iterable[np.ndarray],
                        Y: Union[None, List[str]],
                        weights: Union[None, List[float]] = None,
                        device: torch.device = torch.device("cpu"),
                        pinned: bool = False) -> tuple:
        x = images_to_tensor(X, device, pinned)
        y = None if Y is None else segmented_to_tensor(Y, self.module.colors,
                                                       device, pinned)
        w = None if weights is None else floats_to_tensor(weights, device,
                                                          pinned)
        return x, y, w

    def _tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        indexes = tensor_to_index(tensor)
        return np.array(self.module.colors)[indexes]
