import torch
from typing import Tuple
from .padding import ConstantPad1d, ConstantPad2d, load_padding
from .convolution import Conv1d, Conv2d, load_convolution
from .activation import Activation
from .batch_norm import BatchNorm1d, BatchNorm2d, load_batchnorm
from .pooling import AvgPool1d, AvgPool2d, load_pooling


def load_activatedconv(dump: dict):
    """returns the ConvStage object loaded from a dump"""
    return globals()[dump["type"]].from_dump(dump)


class _ConvStage(torch.nn.Module):
    """A template for activated convolutions"""

    @classmethod
    def from_dump(cls, dump: dict) -> '_ConvStage':
        """returns a '_ConvStageolution' layer from a dump"""
        assert dump["type"] == cls.__name__
        obj = cls(1)
        obj.padded = dump["padded"]
        obj.stacked = dump["stacked"]
        obj.padding = load_padding(dump["padding"])
        obj.convolution = load_convolution(dump["convolution"])
        obj.activation = Activation.from_dump(dump["activation"])
        obj.batch_norm = load_batchnorm(dump["batch norm"])
        obj.downsampler = load_pooling(dump["downsampler"])
        return obj

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.padded:
            X = self.padding(X)
        Y = self.batch_norm(self.activation(self.convolution(X)))
        if self.stacked:
            resized = self.downsampler(X)
            Y = torch.cat([resized, Y], dim=1)
        return Y

    def shape_out(self, shape_in: list) -> list:
        shape_out = self.padding.shape_out(shape_in)
        shape_out = self.convolution.shape_out(shape_out)
        if self.stacked:
            shape_out[0] += self.out_channels
        return shape_out

    def shape_in(self, shape_out: list) -> list:
        if self.stacked:
            shape_out[0] -= self.out_channels
        shape_in = self.convolution.shape_in(shape_out)
        shape_in = self.padding.shape_in(shape_in)
        return shape_in

    @property
    def in_channels(self):
        return self.convolution.in_channels

    @property
    def out_channels(self):
        c = self.convolution.out_channels
        if self.stacked:
            c += self.convolution.in_channels
        return c

    @property
    def window(self):
        return self.convolution.kernel_size

    @property
    def stride(self):
        return self.convolution.stride

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "padded": self.padded,
                "stacked": self.stacked,
                "padding": self.padding.dump,
                "convolution": self.convolution.dump,
                "activation": self.activation.dump,
                "batch norm": self.batch_norm.dump,
                "downsampler": self.downsampler.dump}


class ConvStage1d(_ConvStage):

    def __init__(self, in_channels: int,
                 channels: int = 8,
                 window: int = 3,
                 stride: int = 1,
                 activation: str = "relu",
                 padded: bool = True,
                 with_bias: bool = True,
                 stacked: bool = False):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        channels : int
            The number of output channels
        window : int
            The size of the convolved kernel
        stride : int
            The stride used for the convolution
        padded : bool
            Whether to pad so that the output has the same size as input
        with_bias : bool
            Whether to have a bias in the linear equation:
            y = weight*x + bias
        stacked: bool
            If True the input is downsampled and concatenated to the output
        """
        super().__init__()
        self.padded = padded
        self.stacked = stacked
        left = (window-1)//2
        right = window-1 - left
        pad = (left, right) if padded else (0, 0)
        self.padding = ConstantPad1d(pad, value=0.)
        self.convolution = Conv1d(in_channels, channels, window, stride,
                                  with_bias=with_bias)
        self.activation = Activation(activation)
        self.batch_norm = BatchNorm1d(channels)
        self.downsampler = AvgPool1d(window, stride)


class ConvStage2d(_ConvStage):

    def __init__(self, in_channels: int,
                 channels: int = 8,
                 window: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
                 padded: bool = True,
                 with_bias: bool = True,
                 stacked: bool = False):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        channels : int
            The number of output channels
        window : tuple of int
            The (height, width) size of the convolved kernel
        stride : tuple of int
            The (height, width) stride used for the convolution
        padded : bool
            Whether to pad so that the output has the same size as input
        with_bias : bool
            Whether to have a bias in the linear equation:
            y = weight*x + bias
        stacked: bool
            If True the input is downsampled and concatenated to the output
        """
        super().__init__()
        self.padded = padded
        self.stacked = stacked
        left, top = ((w-1)//2 for w in window)
        right, bottom = (w-1 - first for w, first in zip(window, [left, top]))
        pad = (left, right, top, bottom) if padded else (0, 0, 0, 0)
        self.padding = ConstantPad2d(pad, value=0.)
        self.convolution = Conv2d(in_channels, channels, window, stride,
                                  with_bias=with_bias)
        self.activation = Activation(activation)
        self.batch_norm = BatchNorm2d(channels)
        self.downsampler = AvgPool2d(window, stride)
