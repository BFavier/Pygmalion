import torch
from typing import Tuple, Optional
from ._activation import Activation

class ConvBlock(torch.nn.Module):
    """
    A convolution block of the type of ResNet buolding block
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, in_features: int, out_features: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu", batch_norm: bool=True,
                 residuals: bool = True, n_convolutions: int = 1,
                 dropout: Optional[float] = None):
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
        batch_norm : bool
            whether or not to apply batch norm before each
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.shortcut = torch.nn.Conv2d(in_features, out_features, (1, 1), stride) if residuals else None
        self.batch_norm = torch.nn.BatchNorm2d(out_features) if batch_norm else None
        self.activation = Activation(activation)
        self.dropout = None if dropout is None else torch.nn.Dropout2d(dropout)
        for i in range(1, n_convolutions+1):
            self.layers.append(
                torch.nn.Conv2d(in_features, out_features, kernel_size,
                                stride, padding="same"))
            stride = (1, 1)
            in_features = out_features
            if i == n_convolutions:
                break
            if batch_norm:
                self.layers.append(torch.nn.BatchNorm2d(out_features))
            self.layers.append(Activation(activation))

    def forward(self, X):
        input = X
        for layer in self.layers:
            X = layer(X)
        if self.shortcut is not None:
            X = X + self.shortcut(input)
        if self.batch_norm is not None:
            X = self.batch_norm(X)
        X = self.activation(X)
        if self.dropout is not None:
            X = self.dropout(X)
        return X
