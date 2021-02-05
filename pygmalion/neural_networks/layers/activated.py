import torch
import torch.nn.functional as F
from typing import Union, Tuple
from .padding import Padding, Padding1d, Padding2d
from .weighting import Weighting, Linear, Conv1d, Conv2d
from .batch_norm import BatchNorm, BatchNorm1d, BatchNorm2d
from .dropout import Dropout


class Activated(torch.nn.Module):
    """
    An 'Activated' layer is a succession of:
        - Padding (optional, only for 1d/2d case)
        - Weighting (linear for 0d, or convolution for 1d/2d)
        - An activation function
        - Batch normalization
        - Dropout (during training, optional)
    """

    @classmethod
    def from_dump(cls, dump: dict):
        cls = globals()[dump["type"]]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        if "padding" in dump.keys():
            obj.padding = Padding.from_dump(dump["padding"])
        obj.weighting = Weighting.from_dump(dump["weighting"])
        obj.activation = dump["activation"]
        obj.normalization = BatchNorm.from_dump(dump["normalization"])
        obj.dropout = Dropout.from_dump(dump["dropout"])
        obj.stacked = dump["stacked"]
        return obj

    def shape_out(self, shape_in: list) -> list:
        shape_out = shape_in
        if hasattr(self, "padding"):
            shape_out = self.padding.shape_out(shape_in)
        shape_out = self.weighting.shape_out(shape_out)
        return shape_out

    def shape_in(self, shape_out: list) -> list:
        shape_in = self.weighting.shape_in(shape_out)
        shape_in = self.padding.shape_in(shape_in)
        return shape_in

    def out_channels(self, in_channels: int) -> int:
        assert in_channels == self.weighting.in_channels
        c = self.weighting.out_channels
        if self.stacked:
            c += self.weighting.in_channels
        return c

    def in_channels(self, out_channels: int) -> int:
        if self.stacked:
            assert out_channels == (self.weighting.out_channels
                                    + self.weighting.in_channels)
        else:
            assert out_channels == self.weighting.out_channels
        return self.weighting.in_channels

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        input = X
        if hasattr(self, "padding"):
            X = self.padding(X)
        X = self.weighting(X)
        X = self.function(X)
        X = self.normalization(X)
        X = self.dropout(X)
        if self.stacked:
            X = torch.cat([self.downsample(input), X], dim=1)
        return X

    @property
    def dump(self):
        d = {"type": type(self).__name__}
        if hasattr(self, "padding"):
            d["padding"] = self.padding.dump
        d["weighting"] = self.weighting.dump
        d["activation"] = self.activation
        d["normalization"] = self.normalization.dump
        d["dropout"] = self.dropout.dump
        d["stacked"] = self.stacked
        return d

    @property
    def function(self):
        if hasattr(torch, self.activation):
            return getattr(torch, self.activation)
        else:
            return getattr(F, self.activation)


class Activated0d(Activated):

    def __init__(self, in_channels: int,
                 channels: int = 8,
                 activation: str = "relu",
                 dropout: Union[None, float] = None,
                 bias: bool = True,
                 stacked: bool = False):
        super().__init__()
        self.weighting = Linear(in_channels, channels, bias=bias)
        self.normalization = BatchNorm1d(channels)
        self.activation = activation
        self.dropout = Dropout(dropout)
        self.stacked = stacked

    def out_channels(self, in_channels: int) -> int:
        assert in_channels == self.weighting.in_features
        c = self.weighting.out_features
        if self.stacked:
            c += self.weighting.in_features
        return c

    def in_channels(self, out_channels: int) -> int:
        if self.stacked:
            assert out_channels == (self.weighting.out_features
                                    + self.weighting.in_features)
        else:
            assert out_channels == self.weighting.out_features
        return self.weighting.in_channels

    def downsample(self, X: torch.Tensor) -> torch.Tensor:
        return X


class Activated1d(Activated):

    def __init__(self, in_channels: int,
                 channels: int = 8,
                 window: int = 3,
                 stride: int = 1,
                 activation: str = "relu",
                 dropout: Union[None, float] = None,
                 padded: bool = True,
                 bias: bool = True,
                 stacked: bool = False):
        super().__init__()
        if padded:
            left, right = window // 2, window - 1 - window // 2
            self.padding = Padding1d((left, right))
        self.weighting = Conv1d(in_channels, channels, kernel_size=window,
                                stride=stride, bias=bias)
        self.normalization = BatchNorm1d(channels)
        self.activation = activation
        self.dropout = Dropout(dropout)
        self.stacked = stacked

    def downsample(self, X: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "padding"):
            window = self.weighting.kernel_size
            left, right = window // 2, window - window // 2
            X = X[:, :, left:-right]
        if self.weighting.stride != 1:
            X = F.avg_pool1d(X, self.weighting.stride)
        return X


class Activated2d(Activated):

    def __init__(self, in_channels: int,
                 channels: int = 8,
                 window: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
                 dropout: Union[None, float] = None,
                 padded: bool = True,
                 bias: bool = True,
                 stacked: bool = False):
        super().__init__()
        if padded:
            top, left = [w // 2 for w in window]
            bot, right = [w - 1 - w // 2 for w in window]
            self.padding = Padding2d((left, right, top, bot))
        self.weighting = Conv2d(in_channels, channels, kernel_size=window,
                                stride=stride, bias=bias)
        self.normalization = BatchNorm2d(channels)
        self.activation = activation
        self.dropout = Dropout(dropout)
        self.stacked = stacked

    def downsample(self, X: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "padding"):
            window = self.weighting.kernel_size
            top, left = [w // 2 for w in window]
            bot, right = [w - w // 2 for w in window]
            X = X[:, :, top:-bot, left:-right]
        if self.weighting.stride != (1, 1):
            X = F.avg_pool2d(X, self.weighting.stride)
        return X
