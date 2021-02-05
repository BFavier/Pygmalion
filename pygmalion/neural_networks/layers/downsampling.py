import torch
from .pooling import Pooling, Pooling1d, Pooling2d
from .dense import Dense, Dense1d, Dense2d
from typing import Union, List, Tuple


class Downsampling(torch.nn.Module):
    """
    A downsampling layer is a 'DenseNd' layer
    followed by a 'PoolingNd' layer.
    """

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.dense = Dense.from_dump(dump["dense"])
        obj.pooling = Pooling.from_dump(dump["pooling"])

    def __init__(self, in_channels: int,
                 dense_layer: Union[List[dict], dict],
                 pooling_window: Union[int, Tuple[int, int]],
                 pooling_type: str,
                 **kwargs):
        """
        Parameters
        ----------
        in_channels : int
            the number of channels of the input
        dense_layer : dict, or list of dict
            the parameters of all layers of the 'DenseNd'
        pooling_window : int, or tuple of int
            the size of the pooling window
        pooling_type : 'max' or 'avg'
            The type of pooling
        **kwargs
            additional kwargs passed to DenseNd
        """
        super().__init__()
        dense = self.DenseNd(in_channels, dense_layer, **kwargs)
        pooling = self.PoolingNd(pooling_window, pooling_type)
        self.dense = dense
        self.pooling = pooling

    def forward(self, X: torch.tensor) -> torch.Tensor:
        X = self.dense(X)
        X = self.pooling(X)
        return X

    def shape_in(self, shape_out: list) -> list:
        return self.dense.shape_in(self.pooling.shape_in(shape_out))

    def shape_out(self, shape_in: list) -> list:
        return self.pooling.shape_out(self.dense.shape_out(shape_in))

    def in_channels(self, out_channels: int) -> int:
        return self.dense.in_channels(out_channels)

    def out_channels(self, in_channels: int) -> int:
        return self.dense.out_channels(in_channels)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "dense": self.dense.dump,
                "pooling": self.pooling.dump}


class Downsampling1d(Downsampling):

    DenseNd = Dense1d
    PoolingNd = Pooling1d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def downsampling_factor(self) -> int:
        f = 1
        for activated in self.dense:
            f *= activated.weighting.stride
        f *= self.pooling.pooling_window
        return f


class Downsampling2d(Downsampling):

    DenseNd = Dense2d
    PoolingNd = Pooling2d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def downsampling_factor(self) -> int:
        fh, fw = 1, 1
        for activated in self.dense:
            h, w = activated.weighting.stride
            fh *= h
            fw *= w
        h, w = self.pooling.pooling_window
        return [fh*h, fw*w]
