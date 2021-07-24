import torch
from typing import Union, List, Tuple
from ._encoder import Encoder1d, Encoder2d
from ._decoder import Decoder1d, Decoder2d


class UNet(torch.nn.Module):
    """
    An UNet structure is a encoder followed by a decoder
    with a skip connection between layers of same depth
    """

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        cls = globals()[dump["type"]]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.encoder = cls.EncoderNd.from_dump(dump["encoder"])
        obj.decoder = cls.DecoderNd.from_dump(dump["decoder"])
        return obj

    def __init__(self, in_features: int,
                 downsampling: List[Union[dict, List[dict]]],
                 pooling: List[Union[int, Tuple[int, int]]],
                 upsampling: List[Union[dict, List[dict]]],
                 pooling_type: str = "max",
                 upsampling_method: str = "nearest",
                 activation: str = "relu",
                 stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        in_features : int
            The number of channels of the input
        downsampling : list of [dict / list of dict]
            the kwargs for the Dense layer for all downsampling layers
        pooling : list of [int / tuple of int]
            the pooling window of all downsampling layers
        upsampling : list of [dict / list of dict]
            the kwargs for the Dense layer for all upsampling layers
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
        self.encoder = self.EncoderNd(in_features, downsampling, pooling,
                                      pooling_type=pooling_type,
                                      padded=True,
                                      stacked=stacked,
                                      activation=activation,
                                      dropout=dropout)
        channels = []
        up_factors = []
        out_features = in_features
        for down in self.encoder.stages:
            channels.insert(0, out_features)
            up_factors.insert(0, down.downsampling_factor)
            out_features = down.out_features(out_features)
        self.decoder = self.DecoderNd(out_features, upsampling, up_factors,
                                      stacked_channels=channels,
                                      upsampling_method=upsampling_method,
                                      padded=True,
                                      stacked=stacked,
                                      activation=activation,
                                      dropout=dropout)

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

    def in_features(self, out_features: int) -> int:
        return self.encoder.in_features(out_features)

    def out_features(self, in_features: int) -> int:
        return self.decoder.out_features(in_features)

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
