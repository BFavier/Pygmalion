import torch as _torch
import numpy as _np
from typing import Dict, Union
from ._conversions import tensor_to_index
from ._neural_network import NeuralNetwork


class NeuralNetworkClassifier(NeuralNetwork):

    @classmethod
    def from_dump(cls, dump: dict) -> 'NeuralNetworkClassifier':
        obj = NeuralNetwork.from_dump.__func__(cls, dump)
        obj.class_weights = dump["class weights"]
        return obj

    def __init__(self, *args, class_weights=None, **kwargs):
        """
        Parameters
        ----------
        *args : tuple
            args passed to the NeuralNetwork constructor
        class_weights : dict or None
            a dict of {class: weight} or None
        **kwargs : dict
            kwargs passed to the NeuralNetwork constructor
        """
        super().__init__(*args, **kwargs)
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = [class_weights.get(c, 0)
                                  for c in self.classes]

    def index(self, X) -> _np.ndarray:
        """
        Returns the category index for each observation

        Parameters
        ----------
        X : Any
            The input X of the model.
            it's type depend on the neural network type.
            see 'help(self.module)'

        Returns
        -------
        np.ndarray :
            An array of long, indexes of the 'self.category'
        """
        x, _, _ = self.module.data_to_tensor(X, None, None)
        return tensor_to_index(self.module(x))

    @property
    def class_weights(self):
        if self.module.class_weights is None:
            return None
        else:
            weights = self.module.class_weights.cpu().tolist()
            classes = self.classes
            return {c: w for c, w in zip(classes, weights)}

    @class_weights.setter
    def class_weights(self, other: Union[Dict[object, float], None]):
        if other is not None:
            other = [other.get(c, 1.) for c in self.classes]
            other = _torch.tensor(other, dtype=_torch.float,
                                  device=self.device)
        self.module.class_weights = other

    @property
    def classes(self):
        return self.module.classes

    @property
    def dump(self):
        d = super().dump
        d["class weights"] = self.class_weights
        return d
