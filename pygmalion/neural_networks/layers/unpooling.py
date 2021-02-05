import torch
import torch.nn.functional as F
from typing import Union, Tuple


class Unpooling(torch.nn.Module):
    """
    An unpooling layer increases spatial dimensions of a feature map
    by upscaling it using linear/bilinear interpolation
    """

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        cls = globals()[dump["type"]]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.factor = tuple(dump["factor"])
        return obj

    def __init__(self, factor: Union[int, Tuple[int]],
                 method: str):
        """
        Parameters:
        -----------
        factor : int, or Tuple of int
            the upsampling factor
        method : one of {'nearest', 'interpolate'}
            the method used to
        """
        assert method in ["nearest", "interpolate"]
        super().__init__()
        self.factor = factor
        self.method = method

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "factor": [f for f in self.factor],
                "method": self.method}


class Unpooling1d(Unpooling):

    def __init__(self, factor: int = 2,
                 method: str = "nearest"):
        super().__init__(factor, method)

    def forward(self, X):
        mode = "linear" if self.method == "interpolate" else "nearest"
        return F.interpolate(X, scale_factor=self.factor,
                             mode=mode)


class Unpooling2d(Unpooling):

    def __init__(self, factor: Tuple[int, int] = (2, 2),
                 method: str = "nearest"):
        super().__init__(factor, method)

    def forward(self, X):
        mode = "bilinear" if self.method == "interpolate" else "nearest"
        align = False if self.method == "interpolate" else None
        return F.interpolate(X, scale_factor=self.factor,
                             mode=mode,
                             align_corners=align)
