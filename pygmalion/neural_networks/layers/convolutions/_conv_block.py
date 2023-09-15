import torch
from typing import Tuple, Optional
from pygmalion.neural_networks.layers import Activation, FeaturesNorm, Dropout2d
from ._padded_conv import PaddedConv2d


class ConvBlock(torch.nn.Module):
    """
    A convolution block with one or more convolutions, and optional shortcut
    """

    def __init__(self, in_features: int, out_features: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu", normalize: bool=True,
                 residuals: bool = True, n_convolutions: int = 1,
                 dropout: Optional[float] = None,
                 intermediate_features: Optional[int] = None):
        """
        Parameters
        ----------
        in_features : int
            number of input channels
        out_features : int
            number of output channels
        kernel_size : tuple of (int, int) or int
            (height, width) of the kernel window in pixels
        stride : tuple of (int, int) or int
            (dy, dx) displacement of the kernel window in the first convolution
        activation : str
            name of the activation function
        normalize : bool
            whether or not to apply batch norm before each activation
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout = Dropout2d(dropout)
        self.shortcut = torch.nn.Conv2d(in_features, out_features, (1, 1), stride) if residuals else None
        for i in range(1, n_convolutions+1):
            if (i < n_convolutions):
                features = (intermediate_features or max(in_features, out_features))
            else:
                features = out_features
            self.layers.append(PaddedConv2d(in_features, features, kernel_size, stride))
            if normalize:
                self.layers.append(FeaturesNorm(1, features))
            self.layers.append(Activation(activation))
            stride = (1, 1)
            in_features = features

    def forward(self, X):
        X = X.to(self.device)
        input = X
        for layer in self.layers:
            X = layer(X)
        X = self.dropout(X)
        if self.shortcut is not None:
            X = X + self.shortcut(input)
        return X

    @property
    def device(self) -> torch.device:
        return next(layer for layer in self.modules() if isinstance(layer, torch.nn.Conv2d)).weight.device
