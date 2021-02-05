import torch
from .unpooling import Unpooling, Unpooling1d, Unpooling2d
from .dense import Dense, Dense1d, Dense2d
from typing import Union, List, Tuple


class Upsampling(torch.nn.Module):
    """
    An upsampling layer is an 'UnpoolingNd' layer
    followed by a 'DenseNd' layer.
    """

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.unpooling = Unpooling.from_dump(dump["unpooling"])
        obj.dense = Dense.from_dump(dump["dense"])

    def __init__(self, in_channels: int,
                 dense_layer: Union[List[dict], dict],
                 upsampling_factor: Union[int, Tuple[int, int]],
                 upsampling_method: str = "nearest",
                 **kwargs):
        """
        Parameters
        ----------
        in_channels : int
            the number of channels of the input
        dense_layer : dict, or list of dict
            the parameters of all layers of the 'DenseNd'
        upsampling_factor : int, or tuple of int
            the upsampling factor
        upsampling_method : one of {'nearest', 'interpolate'}
            the method used to unpool
        **kwargs
            additional kwargs passed to DenseNd
        """
        super().__init__()
        unpooling = self.UnpoolingNd(factor=upsampling_factor,
                                     method=upsampling_method)
        dense = self.DenseNd(in_channels, dense_layer, **kwargs)
        self.unpooling = unpooling
        self.dense = dense

    def forward(self, X: torch.tensor,
                Xstack: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Upsample X, optionnaly concatenate Xstack to it,
        then apply a dense layer.

        Parameters:
        -----------
        X : torch.Tensor
            the input of the model
        Xstack : torch.Tensor or None
            if a tensor is provided, the channels of Xstack are concatenated
            to the channels of X after the upsampling layer.
            This is usefull for UNet architectures

        Returns:
        -------
        torch.Tensor :
            result of the layer
        """
        X = self.unpooling(X)
        if Xstack is not None:
            X = self.concat(Xstack, X)
        X = self.dense(X)
        return X

    def shape_in(self, shape_out: list) -> list:
        return self.dense.shape_in(self.pooling.shape_in(shape_out))

    def shape_out(self, shape_in: list) -> list:
        return self.dense.shape_out(self.pooling.shape_out(shape_in))

    def in_channels(self, out_channels: int) -> int:
        return self.dense.in_channels(out_channels)

    def out_channels(self, in_channels: int) -> int:
        return self.dense.out_channels(in_channels)

    def concat(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        return [X1, X2] concatenated along the channel axis
        if X2 is smaller than X1, it is padded with 0
        """
        padding = [[0, l1 - l2] for l1, l2 in
                   zip(X1.shape[2:], X2.shape[2:])]
        padding = sum(padding, [])
        if any(p > 0 for p in padding):
            X2 = torch.nn.functional.pad(X2, padding, value=0.)
        return torch.cat([X1, X2], dim=1)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "unpooling": self.unpooling.dump,
                "dense": self.dense.dump}


class Upsampling1d(Upsampling):

    UnpoolingNd = Unpooling1d
    DenseNd = Dense1d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def factor(self) -> int:
        f = 1
        for activated in self.dense:
            f *= activated.weighting.stride
        f *= self.pooling.pooling_window
        return f


class Upsampling2d(Upsampling):

    UnpoolingNd = Unpooling2d
    DenseNd = Dense2d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def factor(self) -> Tuple[int, int]:
        fh, fw = 1, 1
        for activated in self.dense:
            h, w = activated.weighting.stride
            fh *= h
            fw *= w
        h, w = self.pooling.pooling_window
        return [fh*h, fw*w]
