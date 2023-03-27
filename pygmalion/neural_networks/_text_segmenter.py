import torch
import pandas as pd
from typing import Union, List, Optional, Literal, Iterable
from .layers.transformers import TransformerEncoder, ATTENTION_TYPE
from .layers import LearnedPositionalEncoding, SinusoidalPositionalEncoding
from ._conversions import strings_to_tensor, tensor_to_classes, tensor_to_probabilities
from ._conversions import classes_to_tensor
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy
from pygmalion.tokenizers._utilities import Tokenizer


class TextTranslator(NeuralNetworkClassifier):

    def __init__(self, tokenizer: Tokenizer,
                 classes: Iterable[str],
                 n_stages: int, projection_dim: int, n_heads: int,
                 activation: str = "relu",
                 dropout: Union[float, None] = None,
                 positional_encoding_type: Literal["sinusoidal", "learned", None] = "sinusoidal",
                 mask_padding: bool = True,
                 attention_type: ATTENTION_TYPE = "scaled dot product",
                 RPE_radius: Optional[int] = None,
                 input_sequence_length: Optional[int] = None,
                 low_memory: bool = True):
        """
        Parameters
        ----------
        ...
        """
        super().__init__(classes)
        self.mask_padding = mask_padding
        self.input_sequence_length = input_sequence_length
        embedding_dim = projection_dim*n_heads
        self.tokenizer = tokenizer
        self.embedding = torch.nn.Embedding(self.tokenizer.n_tokens,
                                            embedding_dim)
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        if positional_encoding_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding()
        elif positional_encoding_type == "learned":
            assert input_sequence_length is not None
            self.positional_encoding = LearnedPositionalEncoding(input_sequence_length, embedding_dim)
        elif positional_encoding_type is None:
            self.positional_encoding = None
        else:
            raise ValueError(f"Unexpected positional encoding type '{positional_encoding_type}'")
        self.transformer_encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      RPE_radius=RPE_radius, attention_type=attention_type,
                                                      low_memory=low_memory)
        self.head = torch.nn.Linear(embedding_dim, len(self.classes))

    def forward(self, X: torch.Tensor, padding_mask: Optional[torch.Tensor]):
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
        if self.positional_encoding_input is not None:
            X = self.positional_encoding_input(X)
        if self.dropout_input is not None:
            X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        X = self.transformer_encoder(X, padding_mask)
        return self.head(X)

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
        padding_mask = (x == self.tokenizer.PAD) if self.mask_padding else None
        y_pred = self(x, padding_mask)
        return cross_entropy(y_pred, y_target, weights, class_weights)

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def _x_to_tensor(self, x: List[str],
                     device: Optional[torch.device] = None,
                     max_input_sequence_length: Optional[int] = None,
                     raise_on_longer_sequences: bool = False):
        return strings_to_tensor(x, self.tokenizer, device,
                                 max_sequence_length=self.input_sequence_length or max_input_sequence_length,
                                 raise_on_longer_sequences=raise_on_longer_sequences,
                                 add_start_end_tokens=False)

    def _y_to_tensor(self, y: List[str],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        return classes_to_tensor(y, self.classes, device=device)

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_classes(tensor, self.classes)

    def _tensor_to_proba(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_probabilities(tensor, self.classes)
