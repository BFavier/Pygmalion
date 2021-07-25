import torch
import random
from typing import Union, List, Dict, Tuple, Optional, Callable
from .layers import TransformerEncoder, Embedding
from .layers import Linear, Dropout
from .layers import mask_chronological
from ._conversions import sentences_to_tensor, tensor_to_sentences
from ._conversions import floats_to_tensor
from ._neural_network_classifier import NeuralNetworkClassifier
from ._loss_functions import cross_entropy
from .layers._functional import positional_encoding
from pygmalion.unsupervised import tokenizers
from pygmalion.unsupervised.tokenizers import SpecialToken, Tokenizer
from pygmalion.utilities import document


class TextGeneratorModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.max_length = dump["max length"]
        tkn = getattr(tokenizers, dump["tokenizer"]["type"])
        obj.tokenizer = tkn.from_dump(dump["tokenizer"])
        obj.embedding = Embedding.from_dump(dump["embedding"])
        obj.dropout = Dropout.from_dump(dump["dropout"])
        obj.transformer = TransformerEncoder.from_dump(dump["transformer"])
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self,
                 tokenizer: Tokenizer,
                 n_stages: int,
                 projection_dim: int,
                 n_heads: int,
                 max_length: Optional[int] = None,
                 activation: str = "relu",
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        ...
        """
        super().__init__()
        self.max_length = max_length
        embedding_dim = projection_dim*n_heads
        self.tokenizer = tokenizer
        self.embedding = Embedding(self.tokenizer.n_tokens+3,
                                   embedding_dim)
        self.dropout = Dropout(dropout)
        self.transformer = TransformerEncoder(n_stages, projection_dim,
                                              n_heads,
                                              dropout=dropout,
                                              activation=activation)
        self.output = Linear(embedding_dim,
                             tokenizer.n_tokens+3)

    def complete(self, X: torch.Tensor, p: float = 0.25,
                 max_tokens: int = 100):
        """
        Complete the text
        At each step, the tokens are sorted by probability to be next.
        A random token is chosen amongst the p most probable percentile
        of tokens.
        """
        text_end = self.tokenizer.n_tokens+1
        new_token = None
        while (X.shape[1] < max_tokens
               and new_token != text_end):
            proba = self.next_token(X)
            sorted, indices = torch.sort(proba, descending=True)
            cumulated = torch.cumsum(sorted, dim=-1)
            for i, c in enumerate(cumulated):
                if c >= p:
                    break
            new_token = indices[random.randint(0, i)]
            X = torch.cat([X, torch.full([1, 1], new_token,
                                         dtype=torch.long, device=X.device)],
                          dim=1)
        return X

    def next_token(self, X: torch.Tensor):
        """
        Return the probability of each possible token to be next
        """
        return torch.softmax(self(X)[0, -1, :], dim=-1)

    def forward(self, X):
        N, L = X.shape
        X = self.embedding(X)
        X = positional_encoding(X)
        X = self.dropout(X.reshape(N*L, -1)).reshape(N, L, -1)
        mask = mask_chronological(L, L, X.device)
        X = self.transformer(X, mask)
        X = self.output(X.reshape(N*L, -1)).reshape(N, L, -1)
        return X

    def loss(self, y_pred, y_target, weights=None):
        return cross_entropy(y_pred[:, :-1, :].transpose(1, 2),
                             y_target[:, 1:],
                             weights, self.class_weights)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "max length": self.max_length,
                "tokenizer": self.tokenizer.dump,
                "embedding": self.embedding.dump,
                "dropout": self.dropout.dump,
                "transformer": self.transformer.dump,
                "output": self.output.dump}


class TextGenerator(NeuralNetworkClassifier):

    ModuleType = TextGeneratorModule

    @document(ModuleType.__init__, NeuralNetworkClassifier.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sentence: str = "", p: float = 0.25,
                 max_tokens: int = 100) -> str:
        """
        Complete the given sentence.
        At each step, a random token is added from the p most probable tokens
        """
        self.module.eval()
        x, _, _ = self._data_to_tensor([sentence], None, None,
                                       device=self.device)
        y = self.module.complete(x[:, :-1], p=p, max_tokens=max_tokens)
        return self._tensor_to_y(y)[0]

    def next(self, sentence: str):
        """
        Probability of each token to be next
        """
        X, _, _ = self._data_to_tensor([sentence], None, device=self.device)
        proba = self.module.next_token(X)
        vocab = self.classes
        return {v: p for v, p in zip(vocab, proba)}

    def _data_to_tensor(self, X: List[str], Y: List[str],
                        weights: None = None,
                        device: torch.device = torch.device("cpu")) -> tuple:
        if X is not None:
            x = sentences_to_tensor(X, self.module.tokenizer, device,
                                    max_sequence_length=self.module.max_length)
        else:
            x = None
        if Y is not None:
            y = sentences_to_tensor(Y, self.module.tokenizer, device,
                                    max_sequence_length=self.module.max_length)
        else:
            y = None
        w = None if weights is None else floats_to_tensor(weights, device)
        return x, y, w

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_sentences(tensor, self.module.tokenizer)

    def _batch_generator(self, training_data: Tuple,
                         validation_data: Optional[Tuple],
                         batch_size: Optional[int], n_batches: Optional[int],
                         device: torch.device, shuffle: bool = True
                         ) -> Tuple[Callable, Callable]:
        if self.module.tokenizer.jit:
            generator = self._jit_generator
        else:
            generator = self._static_generator
        training = self._as_generator(training_data*2, generator,
                                      batch_size, n_batches, device, shuffle)
        val = self._as_generator(validation_data*2, self._static_generator,
                                 batch_size, n_batches, device, shuffle)
        return training, val

    @property
    def classes(self):
        start = SpecialToken("START")
        end = SpecialToken("END")
        pad = SpecialToken("PAD")
        return list(self.module.tokenizer.vocabulary) + [start, end, pad]

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
