import torch
import torch.nn.functional as F
from typing import Union, List, Tuple
from .encoder import Encoder, Encoder1d, Encoder2d
from .decoder import Decoder, Decoder1d, Decoder2d


class UNet(torch.nn.Module):
    """
    An UNet structure is a encoder followed by a decoder
    with a skip connection between layers of same depth
    """

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        cls = globals()[dump["type"]]
        return cls.from_dump(dump)

    def __init__(self, in_channels: int,
                 downsampling: List[Union[dict, List[dict]]],
                 pooling: List[Union[int, Tuple[int, int]]],
                 upsampling: List[Union[dict, List[dict]]],
                 pooling_type: str = "max",
                 upsampling_method: str = "nearest",
                 activation: str = "relu",
                 stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        in_channels : int
            The number of channels of the input
        downsampling : list of [dict / list of dict]
            the kwargs for the Dense layer for each
        pooling : list of [dict / list of dict]
            the kwargs for the Dense layer for each
        upsampling : list of [dict / list of dict]
            the kwargs for the Dense layer for each
        pooling_type : one of {"max", "avg"}
            The type of pooling to perform
        upsampling_method : one of {"nearest", "interpolate"}
            The type of pooling to perform
        stacked : bool
            default value for "stacked" in the 'dense_layers' kwargs
        activation : str
            default value for "activation" in the 'dense_layers' kwargs
        dropout : float or None
            default value for "dropout" in the 'dense_layers' kwargs
        """
        assert len(downsampling) == len(pooling) == len(upsampling)
        super().__init__()
        encoder, decoder = [], []
        channels = []
        for dense, pool in zip(downsampling, pooling):
            channels.append(in_channels)
            down = self.DownsamplingNd(in_channels, dense, pool,
                                       pooling_type=pooling_type,
                                       stacked=stacked,
                                       dropout=dropout,
                                       activation=activation,
                                       padded=True)
            in_channels = down.out_channels(in_channels)
            encoder.append(down)
        for dense, down, stack_channels in zip(upsampling, encoder[::-1],
                                               channels[::-1]):
            up = self.UpsamplingNd(in_channels+stack_channels,
                                   dense,
                                   down.downsampling_factor,
                                   upsampling_method=upsampling_method,
                                   stacked=stacked,
                                   dropout=dropout,
                                   activation=activation,
                                   padded=True)
            in_channels = up.out_channels(in_channels+stack_channels)
            decoder.append(up)
        self.encoder = self.EncoderNd.from_layers(encoder)
        self.decoder = self.DecoderNd.from_layers(decoder)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        stack = []
        for stage in self.encoder.stages:
            stack.append(X)
            X = stage(X)
        for stage, Xstack in zip(self.decoder.stages, stack[::-1]):
            X = stage(X, Xstack)
        return X

    def shape_out(self, shape_in: list) -> list:
        shape = shape_in
        shape = self.encoder.shape_out(shape)
        shape = self.decoder.shape_out(shape)
        return shape

    def shape_in(self, shape_out: list) -> list:
        shape = shape_out
        shape = self.decoder.shape_in(shape)
        shape = self.encoder.shape_in(shape)
        return shape

    def out_channels(self, in_channels):
        channels = []
        for stage in self.encoder.stages:
            channels.append(in_channels)
            in_channels = stage.out_channels(in_channels)
        for stage, stack in zip(self.decoder.stages, channels[::-1]):
            in_channels = stage.out_channels(in_channels+stack)
        return in_channels

    @property
    def UpsamplingNd(self):
        return self.DecoderNd.UpsamplingNd

    @property
    def DownsamplingNd(self):
        return self.EncoderNd.DownsamplingNd

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "encoder": self.encoder.dump,
                "decoder": self.decoder.dump}


class UNet1d(UNet):

    EncoderNd = Encoder1d
    DecoderNd = Decoder1d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class UNet2d(UNet):

    EncoderNd = Encoder2d
    DecoderNd = Decoder2d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
