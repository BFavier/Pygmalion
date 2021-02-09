import torch
from typing import Tuple


class Weighting:

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        cls = globals()[dump["type"]]
        return cls.from_dump(dump)


class Linear(torch.nn.Linear, Weighting):
    """A wrapper around torch.nn.Linear"""

    @classmethod
    def from_dump(cls, dump: dict) -> 'Linear':
        """returns a 'Linear' layer from a dump"""
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.in_features = dump["in features"]
        obj.out_features = dump["out features"]
        obj.weight = torch.nn.Parameter(torch.tensor(dump["weight"],
                                                     dtype=torch.float))
        if dump["bias"] is None:
            obj.bias = None
        else:
            obj.bias = torch.nn.Parameter(torch.tensor(dump["bias"],
                                                       dtype=torch.float))
        return obj

    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        Parameters
        ----------
        in_features : int
            The number of input features
        out_features : int
            The number of output features
        bias : bool
            If False, there is not bias term in the linear equation:
            y = weight*x + bias
        """
        super().__init__(in_features, out_features, bias=bias)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "in features": self.in_features,
                "out features": self.out_features,
                "weight": self.weight.tolist(),
                "bias": self.bias if self.bias is None else self.bias.tolist()}

    def shape_out(self, shape_in):
        assert shape_in == []
        return []

    def shape_in(self, shape_out):
        assert shape_out == []
        return []

    @property
    def in_channels(self):
        return self.in_features

    @property
    def out_channels(self):
        return self.out_features


class _Convolution(Weighting):
    """A template for batch norm layers"""

    @classmethod
    def from_dump(cls, dump: dict) -> '_Convolution':
        """returns a '_Convolution' layer from a dump"""
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module__init__(obj)
        obj.in_channels = dump["in channels"]
        obj.out_channels = dump["out channels"]
        obj.kernel_size = dump["kernel size"]
        obj.stride = dump["stride"]
        obj.weight = torch.nn.Parameter(
                        torch.tensor(dump["weight"], dtype=torch.float))
        if dump["bias"] is None:
            obj.bias = None
        else:
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
             "weight": self.weight.tolist()}
        if self.bias is None:
            d["bias"] = None
        else:
            d["bias"] = self.bias.tolist()
        return d

    def shape_out(self, shape_in: list) -> list:
        return [int((d - k)/s + 1) for d, k, s in
                zip(shape_in, self.kernel_size, self.stride)]

    def shape_in(self, shape_out: list) -> list:
        return [(d-1)*s + k for d, k, s in
                zip(shape_out, self.kernel_size, self.stride)]


class Conv1d(torch.nn.Conv1d, _Convolution):
    """A wrapper around torch.nn.Conv1d"""

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 bias: bool = True):
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
        bias : bool
            Whether to have a bias in the linear equation:
            y = weight*x + bias
        """
        torch.nn.Conv1d.__init__(self, in_channels, out_channels,
                                 kernel_size, stride, bias=bias)


class Conv2d(torch.nn.Conv2d, _Convolution):
    """A wrapper around torch.nn.Conv2d"""

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 bias: bool = True):
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
        bias : bool
            Whether to have a bias in the linear equation:
            y = weight*x + bias
        """
        torch.nn.Conv2d.__init__(self, in_channels, out_channels,
                                 kernel_size, stride, bias=bias)
