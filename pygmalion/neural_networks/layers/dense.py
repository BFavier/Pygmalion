import torch
from typing import Union, List
from .activated import Activated, Activated0d, Activated1d, Activated2d


class Dense(torch.nn.Module):
    """
    A Dense layer is a succession of Activated layers
    """

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        cls = globals()[dump["type"]]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.layers = torch.nn.ModuleList()
        for layer in dump["layers"]:
            obj.layers.append(Activated.from_dump(layer))

    def __init__(self, in_channels: int,
                 layers: Union[dict, List[dict]],
                 activation: str = "relu",
                 stacked: bool = False,
                 dropout: Union[None, float] = None,
                 padded: bool = True):
        """
        Parameters
        ----------
        in_channels : int
            the number of channels of the input
        layers : dict, or list of dict
            the parameters of each 'Activated' layer in the dense layer
        activation : str
            default 'activation' parameter for 'ActivatedNd' layers
        stacked : bool
            default 'stacked' parameter for 'ActivatedNd' layers
        dropout : None or float
            default 'dropout' parameter for 'ActivatedNd' layers
        padded : bool
            Only for Dense1d and Dense2d
            default 'padded' parameter for 'ActivatedNd' layers
        """
        super().__init__()
        if isinstance(layers, dict):
            layers = [layers]
        self.layers = torch.nn.ModuleList()
        for kwargs in layers:
            kwargs.setdefault("stacked", stacked)
            kwargs.setdefault("dropout", dropout)
            if type(self) != Dense0d:
                kwargs.setdefault("padded", padded)
            layer = self.ActivatedNd(in_channels, **kwargs)
            in_channels = layer.out_channels(in_channels)
            self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def shape_in(self, shape_out: list) -> list:
        shape = shape_out
        for layer in self.layers[::-1]:
            shape = layer.shape_in(shape)
        return shape

    def shape_out(self, shape_in: list) -> list:
        shape = shape_in
        for layer in self.layers:
            shape = layer.shape_out(shape)
        return shape

    def in_channels(self, out_channels: int) -> int:
        channels = out_channels
        for layer in self.layers[::-1]:
            channels = layer.in_channels(channels)
        return channels

    def out_channels(self, in_channels: int) -> int:
        channels = in_channels
        for layer in self.layers:
            channels = layer.out_channels(channels)
        return channels

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "layers": [layer.dump for layer in self.layers]}


class Dense0d(Dense):

    ActivatedNd = Activated0d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Dense1d(Dense):

    ActivatedNd = Activated1d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Dense2d(Dense):

    ActivatedNd = Activated2d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
