import torch
import torch.nn.functional as F
from typing import Union, Tuple, Literal

UPSAMPLING_METHOD = Literal["nearest", "interpolate"]


class _Upsampling(torch.nn.Module):
    """
    An unpooling layer increases spatial dimensions of a feature map
    by upscaling it using linear/bilinear interpolation
    """

    def __init__(self, factor: Union[int, Tuple[int, int]], method: str):
        """
        Parameters:
        -----------
        factor : int, or Tuple of int
            the upsampling factor
        method : one of {'nearest', 'interpolate'}
            the method used to
        """
        assert method in UPSAMPLING_METHOD.__args__
        super().__init__()
        self.factor = factor
        self.method = method


class Upsampling1d(_Upsampling):

    def __init__(self, factor: int = 2,
                 method: UPSAMPLING_METHOD = "nearest"):
        super().__init__(factor, method)

    def forward(self, X):
        mode = "linear" if self.method == "interpolate" else "nearest"
        align = False if self.method == "interpolate" else None
        return F.interpolate(X, scale_factor=self.factor,
                             mode=mode,
                             align_corners=align)


class Upsampling2d(_Upsampling):

    def __init__(self, factor: Tuple[int, int] = (2, 2),
                 method: UPSAMPLING_METHOD = "nearest"):
        super().__init__(factor, method)

    def forward(self, X):
        mode = "bilinear" if self.method == "interpolate" else "nearest"
        align = False if self.method == "interpolate" else None
        return F.interpolate(X, scale_factor=self.factor,
                             mode=mode,
                             align_corners=align)
