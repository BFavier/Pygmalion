import torch
import torch.nn.functional as F
from typing import Union, List, Tuple
from .pooling_stage import PoolingStage2d, PoolingStage1d, load_poolingstage


class _Encoder(torch.nn.Module):
    """
    An encoder is a succession of PoolingStage
    It reduces spatial dimensions of an image but increase channels depth
    """

    @classmethod
    def from_dump(cls, dump: dict) -> '_Encoder':
        obj = cls(1)
        obj.stages = torch.nn.ModuleList()
        for d in dump["stages"]:
            obj.stages.append(load_poolingstage(d))
        return obj

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            X = stage(X)
        return X

    def shape_out(self, shape_in: List[int]) -> List[int]:
        shape = shape_in
        for stage in self.stages:
            shape = stage.shape_out(shape)
        return shape

    def shape_in(self, shape_out: List[int]) -> List[int]:
        shape = shape_out
        for stage in self.stages[::-1]:
            shape = stage.shape_in(shape)
        return shape

    def shapes(self, shape_in: List[int]) -> List[List[int]]:
        shape = shape_in
        shapes_list = []
        for stage in self.stages:
            shape = stage.shape_out(shape)
            shapes_list.append(shape)
        return shapes_list

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "stages": [s.dump for s in self.stages]}

    @property
    def out_channels(self):
        if len(self.stages) > 0:
            return self.stages[-1].out_channels
        else:
            return self.in_channels


class Encoder1d(_Encoder):

    def __init__(self, in_channels: int,
                 convolution_stages: List[Union[dict, List[dict]]] = [],
                 pooling_windows: List[int] = [],
                 pooling_type: str = "Max",
                 padded: bool = True,
                 stacked: bool = False,
                 activation: str = "relu"):
        """
        in_channels : int
            The number of channels of the input
        convolution_stages : list of dict, or list of list of dict
            the kwargs used to build the successive Convolution1dStage before
            each pooling.
        pooling_windows : list of int
            the window size of the pooling layers
            can be omited if pooling_type is None
        pooling_type : one of {None, "Max", "Avg"}
            The type of pooling to perform
        padded: bool
            default value for the "padded" key in the kwargs
            If omited, the default is used
        stacked: bool
            default value for the "stacked" key in the kwargs
            If omited, the default is used
        activation: str
            default value for the "activation" key in the kwargs
            If omited, the default is used
        """
        super().__init__(in_channels)
        self.stages = torch.nn.ModuleList()
        for conv, pool in zip(convolution_stages, pooling_windows):
            conv.setdefault("padded", padded)
            conv.setdefault("activation", activation)
            cp = PoolingStage1d(in_channels, conv,
                                pooling_type=pooling_type,
                                pooling_size=pool,
                                padded=padded,
                                stacked=stacked,
                                activation=activation)
            self.stages.append(cp)
            in_channels = cp.out_channels


class Encoder2d(_Encoder):

    def __init__(self, in_channels: int,
                 convolution_stages: List[Union[dict, List[dict]]] = [],
                 pooling_windows: List[Tuple[int, int]] = [],
                 pooling_type: str = "Max",
                 padded: bool = True,
                 stacked: bool = False,
                 activation: str = "relu"):
        """
        in_channels : int
            The number of channels of the input
        convolution_stages : list of dict, or list of list of dict
            the kwargs used to build the successive Convolution1dStage before
            each pooling.
        pooling_windows : list of tuple of int
            the (height, width) window size of the pooling layers
            can be omited if pooling_type is None
        pooling_type : one of {None, "Max", "Avg"}
            The type of pooling to perform
        padded: bool
            default value for the "padded" key in the kwargs
            If omited, the default is used
        stacked: bool
            default value for the "stacked" key in the kwargs
            If omited, the default is used
        activation: str
            default value for the "activation" key in the kwargs
            If omited, the default is used
        """
        super().__init__(in_channels)
        self.stages = torch.nn.ModuleList()
        for conv, pool in zip(convolution_stages, pooling_windows):
            cp = PoolingStage2d(in_channels, conv,
                                pooling_type=pooling_type,
                                pooling_size=pool,
                                padded=padded,
                                stacked=stacked,
                                activation=activation)
            self.stages.append(cp)
            in_channels = cp.out_channels
