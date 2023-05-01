import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PaddedConv2d(torch.nn.Module):
    """
    PaddedConv2d is a wrapper around torch.nn.Conv2d that pads the input
    to maintain the same spatial dimension for the output after convolution.
    This layer is memory efficient thanks to gradient checkpointing.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1, 1), dilation=(1, 1), groups=1, bias=True,
                 device=None, dtype=None):
        """
        See torch.nn.Conv2d for parameters documentation
        """
        super().__init__()
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, groups=groups, bias=bias,
            device=device, dtype=dtype)
        kernel_size = tuple(k + (k-1)*(d-1) for k, d in zip(conv.kernel_size, conv.dilation))
        padding = [p for dim in kernel_size[::-1] for p in ((dim-1)//2, (dim-1) - (dim-1)//2)]
        self.padding = torch.nn.ConstantPad2d(padding, 0.)
        self.conv = conv

    def _forward(self, X):
        return self.conv(self.padding(X))

    def forward(self, X: torch.Tensor):
        if self.training and X.requires_grad:
            return checkpoint(self._forward, X)
        else:
            return self._forward(X)



# class AdaptativePad2d(torch.nn.Module):
#     """
#     AdaptativePad2D is a layer that applies zero padding before a convolution
#     in order for the output feature map to have the same (height, width)
#     after the convolution as before padding
#     """

#     def __init__(self, kernel_size: tuple[int, int],
#                  stride: tuple[int, int], dilation: tuple[int, int]):
#         """
#         Parameters
#         ----------
#         kernel_size : tuple of (int, int)
#             the (height, width) of the convolution's convolved kernel
#         stride : tuple of (int, int)
#             the stride along (y, x) of the convolution
#         dilation : tuple of (int, int)
#             the dilation along (y, x) of the convolved kernel
#         """
#         super().__init__()
#         self.kernel = tuple(kernel_size)
#         self.stride = tuple(stride)
#         self.dilation = tuple(dilation)
    
#     def forward(self, X: torch.Tensor):
#         """
#         Parameters
#         ----------
#         X : torch.tensor
#             feature map of shape (N, C, H, W)
        
#         Returns
#         -------
#         torch.Tensor
#             tensor of shape (N, C, H+dh, W+dw) padded with zeros
#         """
#         _, _, h, w = X.shape
#         delta = [((L-1)*s - L + d*(k-1) + 1) for L, k, s, d
#                  in zip((h, w), self.kernel, self.stride, self.dilation)]
#         padding = [p for dim in delta[::-1] for p in (dim//2, dim//2)]
#         if any(p > 0 for p in padding):
#             return F.pad(X, padding, mode="constant", value=0.)
#         else:
#             return X