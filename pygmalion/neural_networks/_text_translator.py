import torch
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Literal
from .layers.transformers import TransformerEncoder, TransformerDecoder, ATTENTION_TYPE
from .layers import LearnedPositionalEncoding, SinusoidalPositionalEncoding
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
                 positional_encoding_type: Literal["sinusoidal", "learned", None] = "sinusoidal",
                 mask_padding: bool = False,
                 attention_type: ATTENTION_TYPE = "scaled dot product",
                 RPE_radius: Optional[int] = None,
                 max_sequence_length: Optional[int] = None,
                 low_memory: bool = True):
        """
        Parameters
        ----------
        ...
        """
        super().__init__()
        self.mask_padding = mask_padding
        self.max_sequence_length = max_sequence_length
        embedding_dim = projection_dim*n_heads
        self.tokenizer_input = tokenizer_input
        self.tokenizer_output = tokenizer_output
        self.embedding_input = torch.nn.Embedding(self.tokenizer_input.n_tokens,
                                                  embedding_dim)
        self.embedding_out = torch.nn.Embedding(self.tokenizer_output.n_tokens,
                                                embedding_dim)
        self.dropout_input = torch.nn.Dropout(dropout) if dropout is not None else None
        self.dropout_output = torch.nn.Dropout(dropout) if dropout is not None else None
        if positional_encoding_type == "sinusoidal":
            self.positional_encoding_input = SinusoidalPositionalEncoding()
            self.positional_encoding_output = SinusoidalPositionalEncoding()
        elif positional_encoding_type == "learned":
            assert max_sequence_length is not None
            self.positional_encoding_input = LearnedPositionalEncoding(max_sequence_length, embedding_dim)
            self.positional_encoding_output = LearnedPositionalEncoding(max_sequence_length, embedding_dim)
        elif positional_encoding_type is None:
            self.positional_encoding_input = None
            self.positional_encoding_output = None
        else:
            raise ValueError(f"Unexpected positional encoding type '{positional_encoding_type}'")
        self.transformer_encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      RPE_radius=RPE_radius, attention_type=attention_type,
                                                      low_memory=low_memory)
        self.transformer_decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      RPE_radius=RPE_radius, attention_type=attention_type,
                                                      low_memory=low_memory)
        self.head = torch.nn.Linear(embedding_dim, self.tokenizer_output.n_tokens)

    def forward(self, X, padding_mask):
        return self.encode(X, padding_mask)

    def encode(self, X: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            tensor of longs of shape (N, L) with:
            * N : number of sentences
            * L : words per sentence
        padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, L)

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, L, D) with D the embedding dimension
        """
        X = X.to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        N, L = X.shape
        X = self.embedding_input(X)
        X = self.positional_encoding_input(X)
        if self.dropout_input is not None:
            X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        X = self.transformer_encoder(X, padding_mask)
        return X

    def decode(self, Y: torch.Tensor, encoded: torch.Tensor, encoded_padding_mask: Optional[torch.Tensor]):
        """
        performs the decoding part of the network

        Parameters
        ----------
        Y : torch.Tensor
            tensor of long of shape (N, Ly) with:
            * N : number of sentences
            * Ly : words per sentence in the output language
        encoded : torch.Tensor
            tensor of floats of shape (N, Lx, D) with:
            * N : number of sentences
            * Lx : words per sentence in the input language
            * D : embedding dim
        encoded_padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, L)

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, Ly, D)
        """
        N, L = Y.shape
        Y = self.embedding_out(Y)
        Y = self.positional_encoding_output(Y)
        if self.dropout_output is not None:
            Y = self.dropout_output(Y.reshape(N*L, -1)).reshape(N, L, -1)
        Y = self.transformer_decoder(Y, encoded, encoded_padding_mask)
        return self.head(Y)

    def loss(self, x, y_target, weights=None, class_weights=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of long of shape (N, Li)
        y_target : torch.Tensor
            tensor of long of shape (N, Lt)
        """
        x, y_target = x.to(self.device), y_target.to(self.device)
        padding_mask = (x == self.tokenizer_input.PAD) if self.mask_padding else None
        encoded = self(x, padding_mask)
        y_pred = self.decode(y_target[:, :-1], encoded, padding_mask)
        return cross_entropy(y_pred.transpose(1, 2), y_target[:, 1:],
                             weights, class_weights)

    def predict(self, X, max_tokens=100):
        START = self.tokenizer_output.PAD
        END = self.tokenizer_output.END
        encoded = self(X)
        # Y is initialized as a single 'start of sentence' character
        Y = torch.full([1, 1], START,
                       dtype=torch.long, device=X.device)
        for _ in range(max_tokens):
            res = self.decode(encoded, Y)
            res = torch.argmax(res, dim=-1)
            index = res[:, -1:]
            Y = torch.cat([Y, index], dim=-1)
            new_token = index.item()
            if new_token == END:
                break
        else:
            Y = torch.cat([Y, index], dim=-1)
        return Y

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def _x_to_tensor(self, x: List[str],
                     device: Optional[torch.device] = None):
        return sentences_to_tensor(x, self.tokenizer_input, device,
                                    max_sequence_length=self.max_sequence_length,
                                    add_start_end_tokens=False)

    def _y_to_tensor(self, y: List[str],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        return sentences_to_tensor(y, self.tokenizer_output, device,
                                   max_sequence_length=self.max_sequence_length,
                                   add_start_end_tokens=True)

    def _tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_sentences(tensor, self.tokenizer_output)
