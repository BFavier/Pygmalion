import torch
import numpy as np
import torch.nn.functional as F
from typing import Union, List, Tuple, Dict, Iterable
from .layers import PoolingStage2d, Linear, BatchNorm2d, Conv2d
from .layers import UNet2d
from .conversions import floats_to_tensor, tensor_to_index
from .conversions import segmented_to_tensor, images_to_tensor
from .nn_decorators import nn_classifier


@nn_classifier
class SemanticSegmenter(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        colors = {cat: col for cat, col in zip(dump["categories"], dump["colors"])}
        obj = cls(1, colors, [], [], [])
        obj.input_norm = BatchNorm2d.from_dump(dump["input norm"])
        obj.UNet = UNet2d.from_dump(dump["U-net"])
        obj.dense = PoolingStage2d.from_dump(dump["dense"])
        obj.output = Conv2d.from_dump(dump["output"])
        return obj

    def __init__(self, in_channels: Tuple[int, int, int],
                 colors: Dict[str, Union[int, List[int]]],
                 downsampling: List[Union[dict, List[dict]]],
                 pooling: List[Tuple[int, int]],
                 upsampling: List[Union[dict, List[dict]]],
                 dense: List[dict] = [],
                 pooling_type: str = "Max",
                 padded: bool = True,
                 activation: str = "relu"):
        super().__init__()
        assert len(downsampling) == len(pooling) == len(upsampling)
        self.categories = [c for c in colors.keys()]
        self.colors = [colors[c] for c in self.categories]
        self.input_norm = BatchNorm2d(in_channels)
        self.UNet = UNet2d(in_channels, downsampling, pooling, upsampling,
                           padded=padded)
        self.dense = PoolingStage2d(self.UNet.out_channels, dense,
                                    pooling_type=None, padded=True)
        self.output = Conv2d(self.dense.out_channels,
                             len(self.categories), (1, 1))

    def forward(self, X: torch.Tensor):
        X = self.input_norm(X)
        X = self.UNet(X)
        X = self.dense(X)
        return self.output(X)

    def data_to_tensor(self, X: Iterable[np.ndarray],
                       Y: Union[None, List[str]],
                       weights: Union[None, List[float]] = None
                       ) -> tuple:
        x = images_to_tensor(X, self.device)
        y = None if Y is None else segmented_to_tensor(Y, self.colors,
                                                       self.device)
        w = None if weights is None else floats_to_tensor(weights)
        return x, y, w

    def tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        indexes = tensor_to_index(tensor)
        return np.array(self.colors)[indexes]

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor,
             weights: Union[torch.Tensor, None]):
        if weights is None:
            return F.cross_entropy(y_pred, y_target)
        else:
            return F.nll_loss(F.log_softmax(y_pred) * weights, y_target)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "categories": self.categories,
                "colors": self.colors,
                "input norm": self.input_norm.dump,
                "U-net": self.UNet.dump,
                "dense": self.dense.dump,
                "output": self.output.dump}
