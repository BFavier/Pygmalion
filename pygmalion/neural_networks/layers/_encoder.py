import torch
from typing import Union, List, Tuple
from ._downsampling import Downsampling, Downsampling1d, Downsampling2d


class Encoder(torch.nn.Module):
    """
    An encoder is a succession of 'DownsamplingNd'
    It reduces spatial dimensions of a feature map
    """

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.stages = torch.nn.ModuleList()
        for d in dump["stages"]:
            obj.stages.append(Downsampling.from_dump(d))
        return obj

    def __init__(self, in_features: int,
                 dense_layers: List[Union[dict, List[dict]]],
                 pooling_windows: List[Union[int, Tuple[int, int]]],
                 pooling_type: str = "max",
                 padded: bool = True,
                 stacked: bool = False,
                 activation: str = "relu",
                 dropout: Union[float, None] = None):
        """
        in_features : int
            The number of channels of the input
        dense_layers : list of [dict / list of dict]
            the kwargs of all 'DenseNd' layers
        pooling_windows : list of int/tuple of int
            the window size of all pooling layers
        pooling_type : one of {"max", "avg"}
            The type of pooling to perform
        padded : bool
            default value for "padded" in the 'dense_layers' kwargs
        stacked : bool
            default value for "stacked" in the 'dense_layers' kwargs
        activation : str
            default value for "activation" in the 'dense_layers' kwargs
        dropout : float or None
            default value for "dropout" in the 'dense_layers' kwargs
        """
        assert len(dense_layers) == len(pooling_windows)
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for dense_layer, pool in zip(dense_layers, pooling_windows):
            stage = self.DownsamplingNd(in_features, dense_layer,
                                        pooling_type=pooling_type,
                                        pooling_window=pool,
                                        padded=padded,
                                        stacked=stacked,
                                        activation=activation,
                                        dropout=dropout)
            self.stages.append(stage)
            in_features = stage.out_features(in_features)

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

    def in_features(self, out_features: int) -> int:
        if len(self.stages) > 0:
            return self.stages[0].in_features(out_features)
        else:
            return out_features

    def out_features(self, in_features: int) -> int:
        if len(self.stages) > 0:
            return self.stages[-1].out_features(in_features)
        else:
            return in_features

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "stages": [s.dump for s in self.stages]}


class Encoder1d(Encoder):

    DownsamplingNd = Downsampling1d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Encoder2d(Encoder):

    DownsamplingNd = Downsampling2d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
