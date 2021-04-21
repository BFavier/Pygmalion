import torch
from typing import Union, List, Iterable
from .layers import TransformerEncoderStage, TransformerDecoderStage
from .layers import Embedding, Linear
from ._conversions import sentences_to_tensor, tensor_to_sentences
from ._conversions import floats_to_tensor
from ._neural_network_classifier import NeuralNetworkClassifier
from ._loss_functions import cross_entropy


class TraductorModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.lexicon_in = dump["lexicon in"]
        obj.lexicon_out = dump["lexicon out"]
        obj.embedding_in = Embedding.from_dump(dump["embedding in"])
        obj.embedding_out = Embedding.from_dump(dump["embedding out"])
        obj.encoders = torch.nn.ModuleList()
        for e in dump["encoders"]:
            obj.encoders.append(TransformerEncoderStage.from_dump(e))
        obj.decoders = torch.nn.ModuleList()
        for d in dump["decoders"]:
            obj.decoders.append(TransformerDecoderStage.from_dump(d))
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self, embedding_dim: int,
                 lexicon_in: Iterable[str],
                 lexicon_out: Iterable[str],
                 projection_dim: int,
                 n_heads: int,
                 layers: List[dict],
                 n_stages: int,
                 activation: str = "relu",
                 stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        in_channels : int
            the number of channels in the input images
        ...
        activation : str
            the default value for the 'activation' key of the kwargs
        stacked : bool
            the default value for the 'stacked' key of the kwargs
        dropout : float or None
            the default value for the 'dropout' key of the kwargs
        """
        super().__init__()
        self.lexicon_in = ["\r"] + list(set(lexicon_in)) + ["\n"]
        self.lexicon_out = ["\r"] + list(set(lexicon_out)) + ["\n"]
        self.embedding_dim = embedding_dim
        self.embedding_in = Embedding(len(self.lexicon_in), embedding_dim)
        self.embedding_out = Embedding(len(self.lexicon_out), embedding_dim)
        self.encoders = torch.nn.ModuleList()
        for stage in range(n_stages):
            t = TransformerEncoderStage(projection_dim, n_heads, layers,
                                        embedding_dim, activation=activation,
                                        stacked=stacked, dropout=dropout)
            self.encoders.append(t)
        self.decoders = torch.nn.ModuleList()
        for stage in range(n_stages):
            t = TransformerDecoderStage(projection_dim, n_heads, layers,
                                        embedding_dim, activation=activation,
                                        stacked=stacked, dropout=dropout)
            self.decoders.append(t)
        self.output = Linear(embedding_dim, len(self.lexicon_out))

    def forward(self, X):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            tensor of longs of shape (N, L) with:
            * N : number of sentences
            * L : words per sentence

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, L, D) with D the embedding dimension
        """
        X = self.embedding_in(X)
        X = self._positional_embedding(X)
        for stage in self.encoders:
            X = stage(X)
        return X

    def decode(self, encoded, Y):
        """
        performs the decoding part of the network

        Parameters
        ----------
        encoded : torch.Tensor
            tensor of floats of shape (N, Lx, D) with:
            * N : number of sentences
            * Lx : words per sentence in the input language
            * D : embedding dim

        Y : torch.Tensor
            tensor of long of shape (N, Ly) with:
            * N : number of sentences
            * Ly : words per sentence in the output language

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, Ly, D)
        """
        Y = self.embedding_out(Y)
        Y = self._positional_embedding(Y)
        for stage in self.decoders:
            Y = stage(encoded, Y)
        return self.output(Y)

    def loss(self, encoded, y_target, weights=None):
        y_pred = self.decode(encoded, y_target[:, :-1])
        return cross_entropy(y_pred.transpose(1, 2), y_target[:, 1:],
                             weights, self.class_weights)

    def predict(self, X, max_words=100):
        encoded = self(X)
        # Y is initialized as a single 'start of sentence' character
        Y = torch.zeros([1, 1], dtype=torch.long, device=X.device)
        for _ in range(max_words):
            res = self.decode(encoded, Y)
            res = torch.argmax(res, dim=-1)
            index = res[:, -1:]
            Y = torch.cat([Y, index], dim=-1)
            if index.item() == len(self.lexicon_out) - 1:
                break
        return Y

    def _positional_embedding(self, X):
        shape = X.shape
        X = X.view(-1, shape[-1])
        N, D = X.shape
        pe = torch.zeros(N, D, dtype=torch.float, device=X.device)
        position = torch.arange(0, D, dtype=torch.float).unsqueeze(0)
        angle = position / 10000**(2*(position//2)/D)
        pe[:, 0::2] = torch.cos(angle[:, 0::2])
        pe[:, 1::2] = torch.sin(angle[:, 1::2])
        X = (X + pe).view(shape)
        return X

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "lexicon in": self.lexicon_in,
                "lexicon out": self.lexicon_out,
                "embedding in": self.embedding_in.dump,
                "embedding out": self.embedding_out.dump,
                "encoders": [e.dump for e in self.encoders],
                "decoders": [d.dump for d in self.decoders],
                "output": self.output.dump}


class Traductor(NeuralNetworkClassifier):

    ModuleType = TraductorModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sentence: str, max_words: int = 100):
        self.module.eval()
        x, _, _ = self._data_to_tensor([sentence], None, device=self.device)
        y = self.module.predict(x)
        return self._tensor_to_y(y)

    def _data_to_tensor(self, X: List[str],
                        Y: Union[None, List[str]],
                        weights: None = None,
                        device: torch.device = torch.device("cpu")) -> tuple:
        x = sentences_to_tensor(X, self.module.lexicon_in, device)
        y = None if Y is None else sentences_to_tensor(Y,
                                                       self.module.lexicon_out,
                                                       device)
        w = None if weights is None else floats_to_tensor(weights, device)
        return x, y, w

    def _tensor_to_y(self, tensor) -> List[str]:
        return tensor_to_sentences(tensor, self.module.lexicon_out)
