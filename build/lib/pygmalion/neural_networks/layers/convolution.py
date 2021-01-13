import torch
from typing import Tuple


def load_convolution(dump: dict):
    """returns the Conv1d or Conv2d object loaded from a dump"""
    return globals()[dump["type"]].from_dump(dump)


class _Convolution:
    """A template for batch norm layers"""

    @classmethod
    def from_dump(cls, dump: dict) -> '_Convolution':
        """returns a '_Convolution' layer from a dump"""
        assert dump["type"] == cls.__name__
        obj = cls(dump["in channels"],
                  dump["out channels"],
                  dump["kernel size"],
                  stride=dump["stride"],
                  with_bias=dump["with bias"])
        obj.weight = torch.nn.Parameter(
                        torch.tensor(dump["weight"], dtype=torch.float))
        if obj.with_bias:
            obj.bias = torch.nn.Parameter(
                        torch.tensor(dump["bias"], dtype=torch.float))
        return obj

    @property
    def dump(self) -> dict:
        d = {"type": type(self).__name__,
             "in channels": self.in_channels,
             "out channels": self.out_channels,
             "kernel size": list(self.kernel_size),
             "stride": list(self.stride),
             "with bias": self.with_bias,
             "weight": self.weight.tolist()}
        if self.with_bias:
            d["bias"] = self.bias.tolist()
        return d

    def shape_out(self, shape_in: list) -> list:
        assert shape_in[0] == self.in_channels
        dims = [int((d - k)/s + 1) for d, k, s in
                zip(shape_in[1:], self.kernel_size, self.stride)]
        return [self.out_channels] + dims

    def shape_in(self, shape_out: list) -> list:
        assert shape_out[0] == self.out_channels
        dims = [(d-1)*s + k for d, k, s in
                zip(shape_out[1:], self.kernel_size, self.stride)]
        return [self.in_channels] + dims


class Conv1d(torch.nn.Conv1d, _Convolution):
    """A wrapper around torch.nn.Conv1d"""

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 with_bias: bool = True):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        out_channels : int
            The number of output channels
        kernel_size : int
            The size of the convolved kernel
        stride : int
            The stride used for the convolution
        with_bias : bool
            Whether to have a bias in the linear equation:
            y = weight*x + bias
        """
        self.with_bias = with_bias
        torch.nn.Conv1d.__init__(self, in_channels, out_channels,
                                 kernel_size, stride, bias=with_bias)


class Conv2d(torch.nn.Conv2d, _Convolution):
    """A wrapper around torch.nn.Conv2d"""

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 with_bias: bool = True):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        out_channels : int
            The number of output channels
        kernel_size : tuple of int
            The (height, width) size of the convolved kernel
        stride : tuple of int
            The (height, width) stride used for the convolution
        with_bias : bool
            Whether to have a bias in the linear equation:
            y = weight*x + bias
        """
        self.with_bias = with_bias
        torch.nn.Conv2d.__init__(self, in_channels, out_channels,
                                 kernel_size, stride, bias=with_bias)
