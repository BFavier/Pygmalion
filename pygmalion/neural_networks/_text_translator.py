import torch
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Callable
from .layers.transformers import TransformerEncoder, TransformerDecoder, ATTENTION_TYPE
from ._conversions import sentences_to_tensor, tensor_to_sentences
from ._conversions import floats_to_tensor
from ._neural_network import NeuralNetwork
from ._loss_functions import cross_entropy
from pygmalion.tokenizers._utilities import Tokenizer, SpecialToken




class TextTranslator(NeuralNetwork):

    def __init__(self, tokenizer_input: Tokenizer,
                 tokenizer_output: Tokenizer,
                 n_stages: int, projection_dim: int, n_heads: int,
                 activation: str = "relu",
                 dropout: Union[float, None] = None,
                 mask_padding: bool = False,
                 attention_type: ATTENTION_TYPE = "scaled dot product",
                 RPE_radius: Optional[int] = None):
        """
        Parameters
        ----------
        ...
        """
        super().__init__()
        embedding_dim = projection_dim*n_heads
        self.tokenizer_input = tokenizer_input
        self.tokenizer_output = tokenizer_output
        self.embedding_in = torch.nn.Embedding(self.tokenizer_input.n_tokens,
                                               embedding_dim)
        self.dropout_in = torch.nn.Dropout(dropout) if dropout is not None else None
        self.transformer_encoder = TransformerEncoder(...)
        self.embedding_out = torch.nn.Embedding(self.tokenizer_output.n_tokens,
                                                embedding_dim)
        self.dropout_out = torch.nn.Dropout(dropout) if dropout is not None else None
        self.transformer_decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation)
        self.output = torch.nn.Linear(embedding_dim, self.tokenizer_output.n_tokens)

    def forward(self, X):
        return self.encode(X)

    def encode(self, X):
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
        N, L = X.shape
        X = self.embedding_in(X)
        X = positional_encoding(X)
        X = self.dropout_in(X.reshape(N*L, -1)).reshape(N, L, -1)
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
        N, L = Y.shape
        Y = self.embedding_out(Y)
        Y = positional_encoding(Y)
        Y = self.dropout_out(Y.reshape(N*L, -1)).reshape(N, L, -1)
        mask = mask_chronological(L, L, Y.device)
        Y = self.transformer.decode(encoded, Y, mask=mask)
        return self.output(Y)

    def loss(self, encoded, y_target, weights=None):
        y_pred = self.decode(encoded, y_target[:, :-1])
        return cross_entropy(y_pred.transpose(1, 2), y_target[:, 1:],
                             weights, self.class_weights)

    def predict(self, X, max_words=100):
        n_tokens = self.tokenizer_out.n_tokens
        sentence_start = n_tokens
        sentence_end = n_tokens+1
        encoded = self(X)
        # Y is initialized as a single 'start of sentence' character
        Y = torch.full([1, 1], sentence_start,
                       dtype=torch.long, device=X.device)
        for _ in range(max_words):
            res = self.decode(encoded, Y)
            res = torch.argmax(res, dim=-1)
            index = res[:, -1:]
            Y = torch.cat([Y, index], dim=-1)
            new_token = index.item()
            if new_token == sentence_end or new_token > n_tokens:
                break
        else:
            Y = torch.cat([Y, index], dim=-1)
        return Y

    def _data_to_tensor(self, X: List[str],
                        Y: Union[None, List[str]],
                        weights: None = None,
                        device: torch.device = torch.device("cpu")) -> tuple:
        if X is not None:
            x = 
        else:
            x = None
        if Y is not None:
            y = 
        else:
            y = None
        w = None if weights is None else floats_to_tensor(weights, device)
        return x, y, w

    def _x_to_tensor(self, x: List[str],
                     device: Optional[torch.device] = None):
        return sentences_to_tensor(x, self.tokenizer_input, device,
                                    max_sequence_length=self.module.max_length)

    def _y_to_tensor(self, y: List[str],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        return sentences_to_tensor(y, self.tokenizer_output, device,
                                   max_sequence_length=self.module.max_length)

    def _tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_sentences(tensor, self.tokenizer_output)
