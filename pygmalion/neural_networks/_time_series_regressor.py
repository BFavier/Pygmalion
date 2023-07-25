import torch
import pandas as pd
from typing import Union, List, Optional, Iterable
from .layers.transformers import TransformerEncoder, ATTENTION_TYPE, FourrierKernelAttention
from .layers import POSITIONAL_ENCODING_TYPE, Dropout
from ._conversions import floats_to_tensor, tensor_to_floats, tensor_to_probabilities
from ._conversions import classes_to_tensor
from ._neural_network import NeuralNetwork
from ._loss_functions import MSE
from pygmalion.tokenizers._utilities import Tokenizer


class TimeSeriesRegressor(NeuralNetwork):

    def __init__(self, inputs: Iterable[str], targets: Iterable[str],
                 n_stages: int, projection_dim: int, n_heads: int,
                 activation: str = "relu",
                 dropout: Union[float, None] = None,
                 gradient_checkpointing: bool = True,
                 positional_encoding_type: Optional[POSITIONAL_ENCODING_TYPE] = None,
                 positional_encoding_kwargs: dict={},
                 attention_type: ATTENTION_TYPE = FourrierKernelAttention,
                 attention_kwargs: dict = {}):
        """
        Parameters
        ----------
        classes : list of str
            the class names
        tokenizer : Tokenizer
            tokenizer of the input sentences
        n_stages : int
            number of stages in the encoder and decoder
        projection_dim : int
            dimension of a single attention head
        n_heads : int
            number of heads for the multi-head attention mechanism
        activation : str
            activation function
        dropout : float or None
            dropout probability if any
        gradient_checkpointing : bool
            If True, uses gradient checkpointing to reduce memory usage during
            training at the expense of computation time.
        positional_encoding_type : POSITIONAL_ENCODING_TYPE or None
            type of absolute positional encoding
        positional_encoding_kwargs : dict
            additional kwargs passed to positional_encoding_type initializer
        attention_type : ATTENTION_TYPE
            type of attention for multi head attention
        attention_kwargs : dict
            additional kwargs passed to attention_type initializer
        """
        super().__init__()
        self.inputs = list(inputs)
        self.targets = list(targets)
        embedding_dim = projection_dim*n_heads
        self.embedding = torch.nn.Linear(len(inputs), embedding_dim)
        self.dropout_input = Dropout(dropout)
        if positional_encoding_type is None:
            self.positional_encoding = None
        else:
            self.positional_encoding = positional_encoding_type(embedding_dim, **positional_encoding_kwargs)
        self.transformer_encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      attention_type=attention_type,
                                                      gradient_checkpointing=gradient_checkpointing, masked=True)
        self.head = torch.nn.Linear(embedding_dim, len(self.targets))

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
            tensor of floats of shape (N, L, C) with C the number of classes
        """
        X = X.to(self.device)
        N, L = X.shape
        X = self.embedding(X)
        if self.positional_encoding is not None:
            X = self.positional_encoding(X)
        X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        X = self.transformer_encoder(X, padding_mask)
        return self.head(X)

    def loss(self, x, padding, y_target, weights=None, class_weights=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of long of shape (N, L)
        y_target : torch.Tensor
            tensor of long of shape (N, L)
        """
        x, y_target = x.to(self.device), y_target.to(self.device)
        y_pred = self(x)
        return MSE(y_pred, y_target, weights, class_weights)

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
