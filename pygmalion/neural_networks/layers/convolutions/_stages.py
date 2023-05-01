import torch
from typing import Tuple, Optional
from ._conv_block import ConvBlock
from ._upsampling import UPSAMPLING_METHOD, Upsampling2d


class ConvolutionalEncoderStage(torch.nn.Module):
    
    def __init__(self, in_features: int,
                 out_features: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 pooling_size: Tuple[int, int] = (2, 2),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
                 n_convs_per_block: int = 1,
                 normalize: bool = True,
                 residuals: bool = True,
                 dropout: Optional[float] = None):
        super().__init__()
        self.convolutions = ConvBlock(in_features, out_features, kernel_size,
                                      stride, activation, normalize, residuals,
                                      n_convs_per_block, dropout,
                                      intermediate_features=out_features)
        self.downsampling = torch.nn.MaxPool2d(pooling_size)
    
    def forward(self, X):
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (N, Cin, Hin, Win)
        
        Returns
        -------
        tuple of torch.Tensor :
            (Y, map) with map the feature map befotre downsampling and Y the downsampled feature map:
            - Y tensor of shape (N, Cout, Hout, Wout)
            - map tensor of shape (N, Cout, Hin, Win)
        """
        maps = self.convolutions(X)
        return self.downsampling(maps), maps


class ConvolutionalDecoderStage(torch.nn.Module):
    
    def __init__(self, in_features: int,
                 out_features: int,
                 add_features: int = 0,
                 kernel_size: Tuple[int, int] = (3, 3),
                 upsampling_factor: Tuple[int, int] = (2, 2),
                 upsampling_method : UPSAMPLING_METHOD = "nearest",
                 activation: str = "relu",
                 n_convs_per_block: int = 1,
                 normalize: bool = True,
                 residuals: bool = True,
                 dropout: Optional[float] = None):
        super().__init__()
        self.upsampling = Upsampling2d(upsampling_factor, method=upsampling_method)
        self.convolutions = ConvBlock(in_features+add_features, out_features, kernel_size,
                                      (1, 1), activation, normalize, residuals,
                                      n_convs_per_block, dropout,
                                      intermediate_features=in_features)

    def forward(self, X: torch.Tensor, add: Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (N, Cin, Hin, Win)
        
        add : torch.Tensor or None
            additional upsampled features to concatenate
            tensor of shape (N, Cadd, Hout, Wout)
        
        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Cout, Hout, Wout)
        """
        X = self.upsampling(X)
        if add is not None:
            _, _, h, w = X.shape
            X = torch.cat([add[:, :, :h, :w], X], dim=1)
        return self.convolutions(X)
