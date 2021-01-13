import torch
from typing import Tuple, Union
from .convolution import Conv2d
from . import pooling as poo
from .activation import Activation
from .padding import ConstantPad2d
from.batch_norm import BatchNorm2d


class Downsampling(torch.nn.Module):
    """
    A downsampling layer is a sequence of:
    - a 'ConstantPad2d'
    - a 'Conv2d', layers
    - a pooling layer 'MaxPool2d' or 'AvgPool2d'
    - an 'Activation' layer
    - a 'BatchNorm2d' layer

    Attributes
    ----------
    padding : ConstantPad2d or None
        The padding layer
    convolution : Conv2d
        The convolution layers
    pooling : MaxPool2d or AvgPool2d or None
        The pooling layer
    activation : Activation
        The activation layer
    batch_norm : BatchNorm2d
        The batch normalization layer
    concatenate : bool
        Whether the input data should be resized
        and concatenated to the output
    """

    @classmethod
    def from_dump(cls, dump: dict) -> 'Downsampling':
        """returns a 'Downsampling' object from a dump"""
        assert dump["type"] == cls.__name__
        obj = cls(1, 1, (1, 1), (1, 1), None,
                  activation=dump["activation"]["function name"],
                  concatenate=dump["concatenate"])
        if dump["padding"] is not None:
            obj.padding = ConstantPad2d.from_dump(dump["padding"])
        obj.convolutions = Conv2d.from_dump(dump["convolution"])
        if dump["pooling"] is not None:
            pooling_type = getattr(poo, dump["pooling"]["type"])
            obj.pooling = pooling_type(dump["pooling"])
        obj.activation = Activation.from_dump(dump["activation"])
        obj.batch_norm = BatchNorm2d.from_dump(dump["batch norm"])
        return obj

    def __init__(self, in_channels: int,
                 out_channels: int,
                 conv_size: Tuple[int, int],
                 conv_stride: Tuple[int, int],
                 pooling_size: Union[None, Tuple[int, int]],
                 activation: str = "relu",
                 padded: bool = True,
                 pooling_kind: str = "Max",
                 concatenate: bool = False):
        """
        Parameters
        ----------
        in_channels : int
            The input channels
        out_channels : int
            The output channels of the successive convolutions
        conv_size : tuple of int
            The (height, width) of convolution window
        conv_stride : tuple of int
            The (height, width) stride of the convolution
        pooling_size : tuple of int or None
            The shape of the pooling window
            If None, no pooling is performed
        activation : str
            The name of the activation function
        padded : bool
            Whether padding should be applied before each convolution.
            If True, padding size is set to keep (height, width) constant
            after each convolution.
        pooling_kind : str
            The type of pooling layer.
            Must be one of ("Max", "Avg")
        concatenate : bool
            Whether the input data should be resized
            and concatenated to the output
        """
        super().__init__()
        if padded:
            h, w = conv_size
            dh, dw = h//2, w//2
            self.padding = ConstantPad2d((dw, dw, dh, dh), 0.)
        else:
            self.padding = None
        self.convolution = Conv2d(in_channels, out_channels,
                                  conv_size, conv_stride)
        if pooling_size is None:
            self.pooling = None
        else:
            self.pooling = getattr(poo, pooling_kind+"Pool2d")(pooling_size)
        self.activation = Activation(activation)
        self.batch_norm = BatchNorm2d(out_channels)
        self.concatenate = concatenate

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        X = input
        if self.padding is not None:
            X = self.padding(X)
        X = self.convolution(X)
        if self.pooling is not None:
            X = self.pooling(X)
        X = self.batch_norm(self.activation(X))
        if self.concatenate:
            Y = input
            # Triming the sides of the input
            if self.padding is None:
                h, w = self.convolution.kernel_size
                dh, dw = h//2, w//2
                Y = Y[:, :, dh:-dh, dw:-dw]
            # Average pool the input to resize to same size as the output
            h, w = self.convolution.kernel_size
            if self.pooling is not None:
                hp, wp = self.pooling.kernel_size
                h, w = h*hp, w*wp
            Y = torch.nn.functional.avg_pool2d(Y, (h, w))
            # Concatenate over the X
            X = torch.cat((Y, X), dim=1)
        # Returns the resulting tensor
        return X

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "padding": None if self.pading is None else self.padding.dump,
                "convolution": self.convolution.dump,
                "pooling": None if self.pooling is None else self.pooling.dump,
                "activation": self.activation.dump,
                "batch norm": self.batch_norm.dump,
                "concatenate": self.concatenate}
