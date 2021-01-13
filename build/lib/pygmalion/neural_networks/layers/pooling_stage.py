import torch
from typing import Union, List, Tuple
from .convolution_stage import ConvStage1d, ConvStage2d, \
                                load_activatedconv
from .operations import linear_interpolation, bilinear_interpolation
from . import pooling as pool


def load_poolingstage(dump: dict):
    return globals()[dump["type"]].from_dump(dump)


class _PoolingStage(torch.nn.Module):
    """
    A succession of activated convolutions
    followed by an optional pooling operation
    """

    @classmethod
    def from_dump(cls, dump):
        assert dump["type"] == cls.__name__
        obj = cls(dump["in channels"])
        obj.pooled = dump["pooled"]
        obj.convolutions = torch.nn.ModuleList()
        obj.convolutions.extend([load_activatedconv(d)
                                 for d in dump["convolutions"]])
        obj.pooling = pool.load_pooling(dump["pooling"])
        return obj

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        X = input
        for conv in self.convolutions:
            X = conv(X)
        if self.pooled:
            X = self.pooling(X)
        return X

    def shape_in(self, shape_out: list) -> list:
        shape = self.pooling.shape_in(shape_out)
        for conv in self.convolutions[::-1]:
            shape = conv.shape_in(shape)
        return shape

    def shape_out(self, shape_in: list) -> list:
        shape = shape_in
        for conv in self.convolutions:
            shape = conv.shape_out(shape)
        return self.pooling.shape_out(shape)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "in channels": self.in_channels,
                "pooled": self.pooled,
                "convolutions": [c.dump for c in self.convolutions],
                "pooling": self.pooling.dump}

    @property
    def out_channels(self):
        if len(self.convolutions) == 0:
            return self.in_channels
        return self.convolutions[-1].out_channels


class PoolingStage1d(_PoolingStage):

    def __init__(self, in_channels,
                 convolutions: Union[dict, List[dict]] = dict(),
                 pooling_type: Union[None, str] = None,
                 pooling_size: int = 2,
                 padded: bool = True,
                 stacked: bool = False,
                 activation: str = "relu"):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        convolutions : list of dict or list of list of dict
            The {"channels", "window", "stride", "bias", "stacked"} kwargs
            used to create each convolution
        pooling_type : one of {"Max", "Avg"}
            The type of pooling to perform after each convolution stage
        pooling_size : tuple of int
            the (height, width) of the pooling window
        default_padded : bool
            If "padded" is omited in a convolution kwarg
            use this value as default
        stacked: bool
            default value for the "stacked" key in the kwargs
            If omited, the default is used
        default_activation : str
            if "activation" is omited in a convolution kwarg
            use this as default
        """
        super().__init__()
        if isinstance(convolutions, dict):
            convolutions = [convolutions]
        self.in_channels = in_channels
        self.pooled = (pooling_type is not None)
        if pooling_type is None:
            pooling_type = "Max"
            pooling_size = 1
        self.convolutions = torch.nn.ModuleList()
        for kwargs in convolutions:
            kwargs.setdefault("padded", padded)
            kwargs.setdefault("stacked", stacked)
            kwargs.setdefault("activation", activation)
            conv = ConvStage1d(in_channels, **kwargs)
            self.convolutions.append(conv)
            in_channels = conv.out_channels
        cls = getattr(pool, pooling_type+"Pool1d")
        self.pooling = cls(kernel_size=pooling_size, stride=pooling_size)

    def upsample(self, X):
        L = X.shape[2]
        _, h, w = self.shape_in([self.out_channels, L])
        return linear_interpolation(X, size=(h, w))


class PoolingStage2d(_PoolingStage):

    def __init__(self, in_channels,
                 convolutions: Union[dict, List[dict]] = dict(),
                 pooling_type: Union[None, str] = None,
                 pooling_size: Tuple[int, int] = (2, 2),
                 padded: bool = True,
                 stacked: bool = False,
                 activation: str = "relu"):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        convolutions : list of dict or list of list of dict
            The {"channels", "window", "stride", "bias", "stacked"} kwargs
            used to create each convolution
        pooling_type : one of {"Max", "Avg"}
            The type of pooling to perform after each convolution stage
        pooling_size : tuple of int
            the (height, width) of the pooling window
        padded : bool
            If "padded" is omited in a convolution kwarg
            use this value as default
        stacked: bool
            default value for the "stacked" key in the kwargs
            If omited, the default is used
        activation : str
            if "activation" is omited in a convolution kwarg
            use this as default
        """
        super().__init__()
        if isinstance(convolutions, dict):
            convolutions = [convolutions]
        self.in_channels = in_channels
        self.pooled = (pooling_type is not None)
        if pooling_type is None:
            pooling_type = "Max"
            pooling_size = (1, 1)
        self.convolutions = torch.nn.ModuleList()
        for kwargs in convolutions:
            kwargs.setdefault("padded", padded)
            kwargs.setdefault("stacked", stacked)
            kwargs.setdefault("activation", activation)
            conv = ConvStage2d(in_channels, **kwargs)
            self.convolutions.append(conv)
            in_channels = conv.out_channels
        cls = getattr(pool, pooling_type+"Pool2d")
        self.pooling = cls(kernel_size=pooling_size, stride=pooling_size)

    def upsample(self, X):
        h, w = X.shape[2:]
        _, h, w = self.shape_in([self.out_channels, h, w])
        return bilinear_interpolation(X, size=(h, w))
