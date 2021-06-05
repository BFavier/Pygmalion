import torch
from typing import Union, List, Dict
from .layers import Transformer, Embedding
from .layers import Linear, Pooling1d
from ._conversions import sentences_to_tensor, tensor_to_classes
from ._conversions import floats_to_tensor, classes_to_tensor
from ._neural_network_classifier import NeuralNetworkClassifier
from ._loss_functions import cross_entropy
from pygmalion.unsupervised.tokenizers import DynamicTokenizer, Tokenizer
from pygmalion.unsupervised.tokenizers import SpecialToken, DynamicTextDataset
from pygmalion.utilities import document


class TextClassifierModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.lexicon_in = dump["lexicon in"]
        obj.lexicon_out = dump["lexicon out"]
        obj.embedding_in = Embedding.from_dump(dump["embedding in"])
        obj.embedding_out = Embedding.from_dump(dump["embedding out"])
        obj.transformer = Transformer.from_dump(dump["transformer"])
        return obj

    def __init__(self,
                 tokenizer: Tokenizer,
                 classes: List[str],
                 n_stages: int,
                 projection_dim: int,
                 n_heads: int,
                 hidden_layers: List[dict],
                 max_length: int = 256,
                 activation: str = "relu",
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        ...
        """
        super().__init__()
        self.max_length = max_length
        self.embedding_dim = projection_dim*n_heads
        self.tokenizer = tokenizer
        self.embedding = Embedding(self.tokenizer.n_tokens+3,
                                   self.embedding_dim)
        self.classes = list(classes)
        self.transformer = Transformer(n_stages, projection_dim, n_heads,
                                       hidden_layers, activation=activation,
                                       dropout=dropout)
        self.pooling = Pooling1d(None)
        self.output = Linear(self.embedding_dim,
                             len(self.classes))

    def forward(self, X):
        X = self._as_tensor(X)
        X = self.embedding(X)
        X = self.transformer(X)
        X = self.pooling(X.transpose(1, 2))
        X = self.output(X)
        return X

    def loss(self, y_pred, y_target, weights=None):
        y_target = self._as_tensor(y_target)
        return cross_entropy(y_pred, y_target,
                             weights, self.class_weights)

    def _as_tensor(self, X: Union[torch.Tensor, DynamicTextDataset]):
        """Converts to tensor if X is a DynamicTextDataset"""
        if issubclass(type(X), DynamicTextDataset):
            X = X.as_tensor(self.training, self.max_length)
        return X

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "embedding in": self.embedding_in.dump,
                "embedding out": self.embedding_out.dump,
                "transformer": self.transformer.dump,
                "output": self.output.dump}


class TextClassifier(NeuralNetworkClassifier):

    ModuleType = TextClassifierModule

    @document(ModuleType.__init__, NeuralNetworkClassifier.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _data_to_tensor(self, X: List[str],
                        Y: Union[None, List[str]],
                        weights: None = None,
                        device: torch.device = torch.device("cpu")) -> tuple:
        x = self._as_trainable(X, self.module.tokenizer, device)
        y = None if Y is None else classes_to_tensor(Y, self.classes,
                                                     device)
        w = None if weights is None else floats_to_tensor(weights, device)
        return x, y, w

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_classes(tensor, self.module.classes)

    def _as_trainable(self, sentences: List[str], tokenizer: Tokenizer, device
                      ) -> object:
        """
        Returns sentences as a DynamicTextDataset or torch.Tensor
        """
        if sentences is None:
            return None
        elif issubclass(type(tokenizer), DynamicTokenizer):
            return DynamicTextDataset(sentences, tokenizer, device)
        else:
            return sentences_to_tensor(sentences, tokenizer, device)

    @property
    def class_weights(self):
        return super().class_weights

    @class_weights.setter
    def class_weights(self, other: Union[Dict[object, float], None]):
        pad = SpecialToken("PAD")
        if other is not None:
            other[pad] = 0.
        else:
            other = {pad: 0.}
        NeuralNetworkClassifier.class_weights.fset(self, other)
