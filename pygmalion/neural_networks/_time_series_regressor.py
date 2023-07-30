import torch
import pandas as pd
from typing import Union, Optional, Iterable
from .layers.transformers import TransformerEncoder, ATTENTION_TYPE, FourrierKernelAttention
from .layers.positional_encoding import POSITIONAL_ENCODING_TYPE
from .layers import Dropout, Normalizer
from ._conversions import named_to_tensor, tensor_to_dataframe
from ._neural_network import NeuralNetwork
from ._loss_functions import MSE


class TimeSeriesRegressor(NeuralNetwork):

    def __init__(self, inputs: Iterable[str], targets: Iterable[str],
                 observation_column: str, time_column: Optional[str],
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
        self.observation_column = observation_column
        self.time_column = str(time_column) if time_column is not None else None
        embedding_dim = projection_dim*n_heads
        self.input_normalizer = Normalizer(-1, len(inputs)) if normalize else None
        self.target_normalizer = Normalizer(-1, len(targets)) if normalize else None
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

    def forward(self, X: torch.Tensor, T: Optional[torch.Tensor], padding_mask: Optional[torch.Tensor], initial_embedding: bool=False):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (N, L, D)
        T : torch.Tensor or None
            tensor of floats of shape (N, L)
        padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, L)

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, L, D)
        """
        X = X.to(self.device)
        if T is not None:
            T = T.to(self.device)
        if self.input_normalizer is not None:
            X = self.input_normalizer(X)
        X = self.embedding(X)
        if initial_embedding:
            X = torch.concatenate([self.initial_embedding.expand(len(X), -1, -1), X], dim=1)
        N, L, _ = X.shape
        if self.positional_encoding is not None:
            X = self.positional_encoding(X)
        X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        attention_kwargs = {"query_positions": T, "key_positions": T} if T is not None else {}
        X = self.transformer_encoder(X, padding_mask, attention_kwargs=attention_kwargs)
        return self.head(X)

    def loss(self, x: torch.Tensor, t: Optional[torch.Tensor], padding_mask: torch.Tensor,
             y_target: torch.Tensor, weights: Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of floats of shape (N, L, D)
        t : torch.Tensor or None
            if provided, the time as a tensor of floats of shape (N, L)
        padding_mask : torch.Tensor
            tensor of booleans of shape (N, L)
        y_target : torch.Tensor
            tensor of floats of shape (N, L, D)
        """
        x, y_target = x.to(self.device), y_target.to(self.device)
        if t is not None:
            t = t.to(self.device)
        y_pred = self(x[:, :-1, :], t, padding_mask, initial_embedding=True)
        if self.target_normalizer is not None:
            y_target = self.target_normalizer(y_target)
        return MSE(y_pred, y_target, weights)

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def data_to_tensor(self, df: pd.DataFrame,
                       device: Optional[torch.device] = None,
                       padded_sequence_length: Optional[int] = None,
                       raise_on_longer_sequences: bool = False) -> tuple:
        X, T, padding_mask = self._x_to_tensor(df, device, padded_sequence_length, raise_on_longer_sequences)
        Y = self._y_to_tensor(df, device, padded_sequence_length)
        return X, T, padding_mask, Y

    def _x_to_tensor(self, df: pd.DataFrame, device: Optional[torch.device] = None,
                     padded_sequence_length: Optional[int] = None,
                     raise_on_longer_sequences: bool = False):
        if raise_on_longer_sequences and padded_sequence_length is not None:
            for obs, x in df.groupby(self.observation_column):
                if len(x) > padded_sequence_length:
                    raise RuntimeError(f"Found sequence longer than {padded_sequence_length} for observation '{obs}'")
        Xs = [named_to_tensor(x, self.inputs) for _, x in df.groupby(self.observation_column)]
        if padded_sequence_length is None:
            padded_sequence_length = max(len(x) for x in Xs)
        X = torch.stack([torch.cat([x, torch.full([padded_sequence_length-len(x), len(self.targets)], float("nan"))])
                         for x in Xs if len(x) <= padded_sequence_length], dim=0)
        padding_mask = torch.stack([(torch.arange(padded_sequence_length) < len(x))
                                    for x in Xs if len(x) <= padded_sequence_length], dim=0)
        if self.time_column is not None:
            Ts = [named_to_tensor(x, [self.time_column])
                for _, x in df.groupby(self.observation_column)]
            T = torch.stack([torch.cat([t, torch.full([padded_sequence_length-len(t), 1], float("nan"))])
                             for t in Ts if len(t) <= padded_sequence_length], dim=0)
        else:
            T = None
        if device is not None:
            X = X.to(device)
            T = T.to(device)
            padding_mask = padding_mask.to(device)
        return X, T, padding_mask

    def _y_to_tensor(self, df: pd.DataFrame, device: Optional[torch.device] = None,
                     padded_sequence_length: Optional[int] = None) -> torch.Tensor:
        Ys = [named_to_tensor(y, self.targets) for _, y in df.groupby(self.observation_column)]
        if padded_sequence_length is None:
            padded_sequence_length = max(len(y) for y in Ys)
        Y = torch.stack([torch.cat([y, torch.full([padded_sequence_length-len(y), len(self.targets)], float("nan"))])
                         for y in Ys if len(y) <= padded_sequence_length], dim=0)
        if device is not None:
            Y = Y.to(device)
        return Y

    def _tensor_to_y(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_dataframe(tensor, self.targets)
