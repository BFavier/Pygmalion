import torch
import torch.nn.functional as F
from typing import Union, List, Tuple
from .pooling_stage import PoolingStage2d, PoolingStage1d, load_poolingstage


class _UNet(torch.nn.Module):
    """
    An UNet structure is a encoder followed by a decoder
    with a skip connection between layers of same depth
    """

    @classmethod
    def from_dump(cls, dump: dict) -> '_UNet':
        obj = cls(dump["in channels"])
        obj.upward = torch.nn.ModuleList()
        for d in dump["upward"]:
            pooling_stage = load_poolingstage(d)
            obj.upward.append(pooling_stage)
        obj.downward = torch.nn.ModuleList()
        for d in dump["downward"]:
            pooling_stage = load_poolingstage(d)
            obj.downward.append(pooling_stage)
        return obj

    def __init__(self, PoolingStage: type,
                 in_channels: int,
                 downward: List[Union[dict, List[dict]]],
                 pooling: List[int],
                 upward: List[Union[dict, List[dict]]],
                 pooling_type: str = "Max",
                 padded: bool = True,
                 activation: str = "relu"):
        super().__init__()
        assert len(downward) == len(upward) > 0
        self.in_channels = in_channels
        self.downward = torch.nn.ModuleList()
        downward_channels = []
        for down, pool in zip(downward, pooling):
            downward_channels.append(in_channels)
            cp = PoolingStage(in_channels, down,
                              pooling_type=pooling_type,
                              pooling_size=pool,
                              padded=padded,
                              activation=activation)
            self.downward.append(cp)
            in_channels = cp.out_channels
        self.upward = torch.nn.ModuleList()
        for up, down_channels in zip(upward, downward_channels[::-1]):
            cp = PoolingStage(in_channels+down_channels, up,
                              pooling_type=None,
                              padded=True,
                              activation=activation)
            self.upward.append(cp)
            in_channels = cp.out_channels

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        results = []
        for stage in self.downward:
            results.append(X)
            X = stage(X)
        for upward, downward, Y in zip(self.upward, self.downward[::-1],
                                       results[::-1]):
            X = self.align(Y, downward.upsample(X))
            X = upward(X)
        return X

    def shape_out(self, shape_in: list) -> list:
        shape = shape_in
        for stage in self.upward:
            shape = stage.shape_out(shape)
        return shape

    def shape_in(self, shape_out: list) -> list:
        shape = shape_out
        for stage in self.stages[::-1]:
            shape = stage.shape_in(shape)
        return shape

    @property
    def out_channels(self):
        if len(self.upward) > 0:
            return self.upward[-1].out_channels
        else:
            return self.in_channels

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "in channels": self.in_channels,
                "downward": [d.dump for d in self.downward],
                "upward": [d.dump for d in self.upward]}


class UNet1d(_UNet):

    def __init__(self, *args, **kwargs):
        super().__init__(PoolingStage1d, *args, **kwargs)

    def align(self, X1: torch.Tensor, X2: torch.Tensor):
        delta = X1.shape[2]-X2.shape[2]
        if delta != 0:
            X2 = F.pad(X2, (0, delta), mode="constant", value=0.)
        return torch.cat([X1, X2], dim=1)


class UNet2d(_UNet):

    def __init__(self, *args, **kwargs):
        super().__init__(PoolingStage2d, *args, **kwargs)

    def align(self, X1: torch.Tensor, X2: torch.Tensor):
        dh = X1.shape[2]-X2.shape[2]
        dw = X1.shape[3]-X2.shape[3]
        if (dh, dw) != (0, 0):
            X2 = F.pad(X2, (0, dw, 0, dh), mode="constant", value=0.)
        return torch.cat([X1, X2], dim=1)
