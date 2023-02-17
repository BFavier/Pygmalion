import torch
import torch.nn.functional as F
from typing import Union, Tuple, Literal


METHOD = Literal["nearest", "bilinear"]

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
        factor = dump["factor"]
        if hasattr(factor, "__iter__"):
            factor = tuple(factor)
        obj.factor = factor
        obj.method = dump["method"]
        return obj

    def __init__(self, factor: Union[int, Tuple[int, int]], method: str):
        """
        Parameters:
        -----------
        factor : int, or Tuple of int
            the upsampling factor
        method : one of {'nearest', 'interpolate'}
            the method used to
        """
        assert method in METHOD.__args__
        super().__init__()
        self.factor = factor
        self.method = method


class Unpooling1d(Unpooling):

    def __init__(self, factor: int = 2,
                 method: METHOD = "nearest"):
        super().__init__(factor, method)

    def forward(self, X):
        mode = "linear" if self.method == "interpolate" else "nearest"
        align = False if self.method == "interpolate" else None
        return F.interpolate(X, scale_factor=self.factor,
                             mode=mode,
                             align_corners=align)


class Unpooling2d(Unpooling):

    def __init__(self, factor: Tuple[int, int] = (2, 2),
                 method: METHOD = "nearest"):
        super().__init__(factor, method)

    def forward(self, X):
        mode = "bilinear" if self.method == "interpolate" else "nearest"
        align = False if self.method == "interpolate" else None
        return F.interpolate(X, scale_factor=self.factor,
                             mode=mode,
                             align_corners=align)
