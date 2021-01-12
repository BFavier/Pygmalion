import torch
from typing import List
from .linear_stage import LinearStage


class FullyConnected(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'FullyConnected':
        obj = cls(1)
        obj.hidden_layers = torch.nn.ModuleList()
        obj.in_features = dump["in features"]
        for d in dump["hidden layers"]:
            obj.hidden_layers.append(LinearStage.from_dump(d))
        return obj

    def __init__(self, in_features: int,
                 hidden_layers: List[int] = [10, 10, 10],
                 activation: str = "relu"):
        super().__init__()
        self.in_features = in_features
        self.hidden_layers = torch.nn.ModuleList()
        for out_features in hidden_layers:
            lin = LinearStage(in_features, out_features, activation)
            self.hidden_layers.append(lin)
            in_features = out_features

    def forward(self, X: torch.Tensor) -> torch.tensor:
        for layer in self.hidden_layers:
            X = layer(X)
        return X

    def shape_in(self, shape_out: int):
        assert shape_out == self.out_channels
        return self.in_features

    def shape_out(self, shape_in: int):
        assert shape_in == self.in_channels
        return self.out_features

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "in features": self.in_features,
                "hidden layers": [h.dump for h in self.hidden_layers]}

    @property
    def out_features(self):
        if len(self.hidden_layers) > 0:
            return self.hidden_layers[-1].out_features
        else:
            return self.in_features
