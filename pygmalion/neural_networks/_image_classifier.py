import torch
import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from .layers import Linear, BatchNorm2d
from .layers import Encoder2d, Dense0d
from ._conversions import floats_to_tensor, tensor_to_index
from ._conversions import classes_to_tensor, images_to_tensor
from ._conversions import tensor_to_probabilities
from ._neural_network_classifier import NeuralNetworkClassifier
from ._loss_functions import cross_entropy


class ImageClassifierModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.classes = dump["classes"]
        obj.input_norm = BatchNorm2d.from_dump(dump["input norm"])
        obj.encoder = Encoder2d.from_dump(dump["encoder"])
        obj.dense = Dense0d.from_dump(dump["dense"])
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self, in_features: int,
                 classes: List[str],
                 convolutions: Union[List[dict], List[List[dict]]],
                 pooling: List[Tuple[int, int]],
                 dense: List[dict],
                 pooling_type: str = "max",
                 padded: bool = True,
                 activation: str = "relu",
                 stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        in_features : int
            the number of channels in the input images
        classes : list of str
            the unique classes the model can predict
        convolutions : list of [dict / list of dict]
            the kwargs for the 'Activated2d' layers for all 'downsampling'
        pooling : list of [int / tuple of int]
            the pooling window of all downsampling layers
            (excepted the last one which is an overall pool)
        dense : list of dict
            the kwargs for the 'Activated0d' of the final 'Dense0d' layer
        pooling_type : one of {'max', 'avg'}
            the type of pooling
        padded : bool
            the default value for the 'padded' key of the kwargs
        activation : str
            the default value for the 'activation' key of the kwargs
        stacked : bool
            the default value for the 'stacked' key of the kwargs
        dropout : float or None
            the default value for the 'dropout' key of the kwargs
        """
        super().__init__()
        assert len(pooling) == len(convolutions) - 1
        self.classes = list(classes)
        self.input_norm = BatchNorm2d(in_features)
        self.encoder = Encoder2d(in_features, convolutions, pooling+[None],
                                 pooling_type=pooling_type,
                                 padded=padded,
                                 activation=activation,
                                 stacked=stacked,
                                 dropout=dropout)
        in_features = self.encoder.out_features(in_features)
        self.dense = Dense0d(in_features, dense, activation=activation,
                             stacked=stacked, dropout=dropout)
        in_features = self.dense.out_features(in_features)
        self.output = Linear(in_features, len(classes))

    def forward(self, X: torch.Tensor):
        X = self.input_norm(X)
        X = self.encoder(X)
        X = self.dense(X)
        return self.output(X)

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor,
             weights: Union[None, torch.Tensor] = None):
        return cross_entropy(y_pred, y_target, weights,
                             self.class_weights)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "classes": list(self.classes),
                "input norm": self.input_norm.dump,
                "encoder": self.encoder.dump,
                "dense": self.dense.dump,
                "output": self.output.dump}


class ImageClassifier(NeuralNetworkClassifier):

    ModuleType = ImageClassifierModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def probability(self, X) -> pd.DataFrame:
        """
        Return the class probabilities for each observation

        Parameters
        ----------
        X : Any
            The input X of the model.
            it's type depend on the neural network type.
            see 'help(self.module)'

        Returns
        -------
        pd.DataFrame :
            DataFrame of class probabilities
            where each column corresponds to a class
        """
        x, _, _ = self.module.data_to_tensor(X, None, None)
        return tensor_to_probabilities(self.module(x), self.classes)

    def _data_to_tensor(self, X: np.ndarray,
                        Y: Union[None, List[str]],
                        weights: Union[None, List[float]] = None,
                        device: torch.device = torch.device("cpu")) -> tuple:
        x = images_to_tensor(X, device)
        y = None if Y is None else classes_to_tensor(Y, self.classes, device)
        w = None if weights is None else floats_to_tensor(weights, device)
        return x, y, w

    def _tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        indexes = tensor_to_index(tensor)
        return [self.classes[i] for i in indexes]
