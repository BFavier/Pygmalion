import torch
import pandas as pd
from typing import Union, List, Optional, Iterable
from .layers.transformers import TransformerEncoder, ATTENTION_TYPE, FourrierKernelAttention
from .layers.positional_encoding import POSITIONAL_ENCODING_TYPE
from .layers import Dropout, Normalizer
from ._conversions import named_to_tensor, tensor_to_dataframe
from ._conversions import classes_to_tensor
from ._neural_network import NeuralNetwork
from ._loss_functions import MSE
from pygmalion.tokenizers._utilities import Tokenizer


class TimeSeriesRegressor(NeuralNetwork):

    def __init__(self, inputs: Iterable[str], targets: Iterable[str],
                 n_stages: int, projection_dim: int, n_heads: int,
                 activation: str = "relu",
                 dropout: Union[float, None] = None,
                 normalize: bool = True,
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
        self.inputs = inputs
        self.targets = targets
        self.input_normalizer = Normalizer(-1, len(inputs), affine=False) if normalize else None
        self.target_normalizer = Normalizer(-1, len(targets), affine=False) if normalize else None
        self.initial_embedding = torch.nn.parameter.Parameter(torch.zeros(1, 1, embedding_dim))
        self.embedding = torch.nn.Linear(len(inputs), embedding_dim)
        self.dropout_input = Dropout(dropout)
        if positional_encoding_type is None:
            self.positional_encoding = None
        else:
            self.positional_encoding = positional_encoding_type(embedding_dim, **positional_encoding_kwargs)
        self.transformer_encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      attention_type=attention_type,
                                                      gradient_checkpointing=gradient_checkpointing,
                                                      **attention_kwargs)
        self.head = torch.nn.Linear(embedding_dim, len(self.targets))

    def forward(self, X: torch.Tensor, T: torch.Tensor, padding_mask: Optional[torch.Tensor], initial_embedding: bool=False):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (N, L, D)
        T : torch.Tensor
            tensor of floats of shape (N, L)
        padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, L)

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, L, D)
        """
        X = X.to(self.device)
        T = T.to(self.device)
        N, L, _ = X.shape
        if self.input_normalizer is not None:
            X = self.input_normalizer(X)
        X = self.embedding(X)
        if initial_embedding:
            X = torch.concatenate([self.initial_embedding, X], dim=1)
        if self.positional_encoding is not None:
            X = self.positional_encoding(X)
        X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        X = self.transformer_encoder(X, padding_mask, attention_kwargs={"query_positions": T, "key_postions": T})
        return self.head(X)

    def loss(self, x, t, padding_mask, y_target, weights=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of floats of shape (N, L, D)
        t : torch.Tensor
        padding_mask : torch.Tensor
            tensor of booleans of shape (N, L)
        y_target : torch.Tensor
            tensor of floats of shape (N, L, D )
        """
        x, t, y_target = x.to(self.device), t.to(self.device), y_target.to(self.device)
        y_pred = self(x[:, :-1, :], t, padding_mask, initial_embedding=True)
        if self.target_normalizer is not None:
            y_target = self.target_normalizer(y_target)
        return MSE(y_pred, y_target, weights)

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def _x_to_tensor(self, x: Union[pd.DataFrame, dict, Iterable],
                     device: Optional[torch.device] = None):
        return named_to_tensor(x, list(self.inputs), device=device)

    def _y_to_tensor(self, y: Union[pd.DataFrame, dict, Iterable],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        return named_to_tensor(y, self.targets, device=device)

    def _tensor_to_y(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_dataframe(tensor, self.targets)
