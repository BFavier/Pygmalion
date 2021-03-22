import torch
from typing import Union, List, Tuple
from ._upsampling import Upsampling, Upsampling1d, Upsampling2d


class Decoder(torch.nn.Module):
    """
    A Decoder is a succession of 'UpsamplingNd' layers.
    It increase the spatial dimensions of a feature map
    """

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.stages = torch.nn.ModuleList()
        for d in dump["stages"]:
            obj.stages.append(Upsampling.from_dump(d))
        return obj

    def __init__(self, in_channels: int,
                 dense_layers: List[Union[dict, List[dict]]],
                 upsampling_factors: List[Union[int, Tuple[int, int]]],
                 stacked_channels: Union[None, List[int]] = None,
                 upsampling_method: str = "nearest",
                 padded: bool = True,
                 stacked: bool = False,
                 activation: str = "relu",
                 dropout: Union[float, None] = None):
        """
        in_channels : int
            The number of channels of the input
        dense_layers : list of [dict / list of dict]
            the kwargs for the Dense layer for each
        upsampling_factors : list of [int / tuple of int]
            The upsampling factor of each layer
        stacked_channels : None or list of int
            The 'stacked_channel' parameter for each upsampling layer
            Used for UNet architectures.
            If None, equivalent to a list full of 0
        upsampling_method : one of {"nearest", "interpolate"}
            The method used to unpool
        padded : bool
            default value for "padded" in the 'dense_layers' kwargs
        stacked : bool
            default value for "stacked" in the 'dense_layers' kwargs
        activation : str
            default value for "activation" in the 'dense_layers' kwargs
        dropout : float or None
            default value for "dropout" in the 'dense_layers' kwargs
        """
        assert len(dense_layers) == len(upsampling_factors)
        if stacked_channels is None:
            stacked_channels = [0]*len(dense_layers)
        else:
            assert len(dense_layers) == len(stacked_channels)
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for d, f, s in zip(dense_layers, upsampling_factors, stacked_channels):
            stage = self.UpsamplingNd(in_channels, d,
                                      upsampling_factor=f,
                                      upsampling_method=upsampling_method,
                                      stacked_channels=s,
                                      padded=padded,
                                      stacked=stacked,
                                      activation=activation,
                                      dropout=dropout)
            self.stages.append(stage)
            in_channels = stage.out_channels(in_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            X = stage(X)
        return X

    def shape_out(self, shape_in: list) -> list:
        for stage in self.stages:
            shape_in = stage.shape_out(shape_in)
        return shape_in

    def shape_in(self, shape_out: list) -> list:
        for stage in self.stages[::-1]:
            shape_out = stage.shape_in(shape_out)
        return shape_out

    def in_channels(self, out_channels: int) -> int:
        for stage in self.stages[::-1]:
            out_channels = stage.in_channels(out_channels)
        return out_channels

    def out_channels(self, in_channels: int) -> int:
        for stage in self.stages:
            in_channels = stage.out_channels(in_channels)
        return in_channels

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "stages": [s.dump for s in self.stages]}


class Decoder1d(Decoder):

    UpsamplingNd = Upsampling1d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Decoder2d(Decoder):

    UpsamplingNd = Upsampling2d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
