import torch
from torch.utils.checkpoint import checkpoint
from typing import Iterable, Tuple, Optional
from itertools import repeat
from ._stages import ConvolutionalEncoderStage, ConvolutionalDecoderStage
from ._upsampling import UPSAMPLING_METHOD


class ConvolutionalEncoder(torch.nn.Module):

    def __init__(self, in_features: int,
                 features: Iterable[int],
                 kernel_size: Tuple[int, int] = (3, 3),
                 pooling_size: Tuple[int, int] = (2, 2),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
                 n_convs_per_block: int = 1,
                 normalize: bool = True,
                 residuals: bool = True,
                 dropout: Optional[float] = None,
                 low_memory: bool = False):
        super().__init__()
        self.low_memory = low_memory
        self.stages = torch.nn.ModuleList()
        for out_features in features:
            self.stages.append(ConvolutionalEncoderStage(
                in_features, out_features, kernel_size, pooling_size, stride,
                activation, n_convs_per_block, normalize, residuals, dropout))
            in_features = out_features

    def forward(self, X: torch.Tensor, maps: Optional[list]=None):
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (N, Cin, Hin, Win)
        
        maps : list or None
            optional list where to store pre-downsampling feature maps
        
        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Cout, Hout, Wout)
        """
        for stage in self.stages:
            if self.low_memory and self.training:
                X.requires_grad_(True)  # To ensure that the gradient is backpropagated in the convolution parameters
                X, map = checkpoint(stage, X)
            else:
                X, map = stage(X)
            if maps is not None:
                maps.append(map)
        return X


class ConvolutionalDecoder(torch.nn.Module):

    def __init__(self, in_features: int,
                 features: Iterable[int],
                 add_features: Optional[Iterable[int]] = None,
                 kernel_size: Tuple[int, int] = (3, 3),
                 upsampling_factor: Tuple[int, int] = (2, 2),
                 upsampling_method : UPSAMPLING_METHOD = "nearest",
                 activation: str = "relu",
                 n_convs_per_block: int = 1,
                 normalize: bool = True,
                 residuals: bool = True,
                 dropout: Optional[float] = None,
                 low_memory: bool = False):
        super().__init__()
        self.low_memory = low_memory
        self.stages = torch.nn.ModuleList()
        for out_features, add in zip(features, add_features or repeat(None)):
            self.stages.append(ConvolutionalDecoderStage(
                in_features, out_features, add, kernel_size, upsampling_factor, upsampling_method,
                activation, n_convs_per_block, normalize, residuals, dropout))
            in_features = out_features

    def forward(self, X: torch.Tensor, add_maps: Optional[Iterable[torch.Tensor]]=None):
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (N, Cin, Hin, Win)
        
        add_maps : iterable of torch.Tensor or None
            additional upsampled features to concatenate
            tensor of shape (N, Cadd, Hout, Wout)
        
        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Cout, Hout, Wout)
        """
        for stage, add in zip(self.stages, add_maps or repeat(None)):
            if self.low_memory and self.training:
                X.requires_grad_(True)  # To ensure that the gradient is backpropagated in the convolution parameters
                X = checkpoint(stage, X, add)
            else:
                X = stage(X, add)
        return X
