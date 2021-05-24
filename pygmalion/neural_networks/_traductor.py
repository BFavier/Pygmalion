import torch
from typing import Union, List
from .layers import Transformer, Embedding
from .layers import Linear
from ._conversions import sentences_to_tensor, tensor_to_sentences
from ._conversions import floats_to_tensor, longs_to_tensor
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
        obj.transformer = Transformer.from_dump(dump["transformer"])
        return obj

    def __init__(self,
                 tokenizer_in,
                 tokenizer_out,
                 n_stages: int,
                 projection_dim: int,
                 n_heads: int,
                 hidden_layers: List[dict],
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
        self.embedding_dim = projection_dim*n_heads
        self.tokenizer_in = tokenizer_in
        self.tokenizer_out = tokenizer_out
        self.embedding_in = Embedding(256, self.embedding_dim)
        self.embedding_out = Embedding(256, self.embedding_dim)
        self.transformer = Transformer(n_stages, projection_dim, n_heads,
                                       hidden_layers, activation=activation,
                                       stacked=stacked, dropout=dropout)
        self.output = Linear(self.embedding_dim, 256)

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
        try:
            X = self.tokenizer_in.encode(X, use_dropout=self.training)
        except TypeError:
            X = self.tokenizer_in.encode(X)
        X = longs_to_tensor(X, self.device)
        X = self.embedding_in(X)
        X = self.transformer.encode(X)
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
        try:
            Y = self.tokenizer_out.encode(Y, use_dropout=self.training)
        except TypeError:
            Y = self.tokenizer_out.encode(Y)
        Y = longs_to_tensor(Y, self.device)
        Y = self.embedding_out(Y)
        Y = self.transformer.decode(encoded, Y)
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
                "embedding in": self.embedding_in.dump,
                "embedding out": self.embedding_out.dump,
                "transformer": self.transformer.dump,
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
        x = X
        y = Y
        w = None if weights is None else floats_to_tensor(weights, device)
        return x, y, w

    def _tensor_to_y(self, tensor) -> List[str]:
        return tensor_to_sentences(tensor, self.module.lexicon_out)


# def sentences_to_tensor(sentences: Iterable[str],
#                         lexicon: List[str],
#                         device: torch.device) -> torch.Tensor:
#     """
#     converts a list of sentences to tensor

#     Parameters
#     ----------
#     sentences : iterable of str
#         a list of sentences: words separated by a single white spaces
#     lexicon : list of str
#         a list of unique possible words

#     Returns
#     -------
#     torch.Tensor :
#         a tensor of shape (N, L) of longs, where:
#         * N is the number of sentences
#         * L is the length of longest sentence
#         and each scalar is the index of a word in the lexicon
#     """
#     assert isinstance(lexicon, list)
#     sentences = [s.split() for s in sentences]
#     L_max = max([len(s) for s in sentences])
#     sentences = [["\r"] + s + ["\n"]*(L_max - len(s) + 1)
#                  for s in sentences]
#     indexes = {c: i for i, c in enumerate(lexicon)}
#     data = [[indexes[w] for w in s] for s in sentences]
#     return longs_to_tensor(data, device)


# def tensor_to_sentences(tensor: torch.Tensor,
#                         lexicon: List[str]) -> List[str]:
#     """
#     converts a tensor to a list of sentences

#     Parameters
#     ----------
#     tensor : torch.Tensor
#         a tensor of shape (N, L) where:
#         * N is the number of sentences
#         * L is the length of longest sentence
#     lexicon : list of str
#         a list of unique possible words

#     Returns
#     -------
#     list of str :
#         a list of sentences,
#         each sentence is a set of words separated by whitespaces
#     """
#     indexes = tensor_to_longs(tensor.view(-1))
#     words = np.array(lexicon)[indexes]
#     sentence = " ".join(words[1:-1])
#     return sentence
