import torch
import pandas as pd
import numpy as np
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
                 std_noise: float = 0.,
                 mask_n_firsts: int = 1,
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
        observation_column : str
            column by which the dataframe will be grouped when grouping observations
        time_column : str or None
            The name of the time column if any. Usefull for non equally spaced time series.
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
        normalize : bool
            if True, the inputs and targets ar enormalized
        std_noise : float
            Standard deviation of normaly distribued noise with zero mean
            added to input during training.
        mask_n_firsts : int
            Number of initial points ignored in the loss during training.
            Must be at least 1.
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
        self.std_noise = std_noise
        self.mask_n_firsts = mask_n_firsts
        embedding_dim = projection_dim*n_heads
        self.input_normalizer = Normalizer(-1, len(inputs)) if normalize else None
        self.target_normalizer = Normalizer(-1, len(targets)) if normalize else None
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
                                                      mask_future=True,
                                                      **attention_kwargs)
        self.head = torch.nn.Linear(embedding_dim, len(self.targets))

    def forward(self, X: torch.Tensor, T: Optional[torch.Tensor], T_next: Optional[torch.Tensor], padding_mask: Optional[torch.Tensor]):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (N, L, D)
        T : torch.Tensor or None
            tensor of floats of shape (N, L)
        T_next : torch.Tensor or None
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
            X = self.input_normalizer(X, padding_mask)
        X = self.embedding(X)
        N, L, _ = X.shape
        if self.positional_encoding is not None:
            X = self.positional_encoding(X)
        X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        attention_kwargs = {"query_positions": T_next, "key_positions": T} if T is not None else {}
        X = self.transformer_encoder(X, padding_mask, attention_kwargs=attention_kwargs)
        return self.head(X)

    def loss(self, x: torch.Tensor, t: Optional[torch.Tensor], padding_mask: Optional[torch.Tensor],
             y_target: torch.Tensor, weights: Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of floats of shape (N, L, D)
        t : torch.Tensor or None
            if provided, the time as a tensor of floats of shape (N, L)
        padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, L)
        y_target : torch.Tensor
            tensor of floats of shape (N, L, D)
        """
        N, L, D = x.shape
        x, y_target = x.to(self.device), y_target.to(self.device)
        if t is not None:
            t = t.to(self.device)
        noise = torch.normal(torch.zeros((N, L-1, D), device=x.device),
                             torch.full((N, L-1, D), self.std_noise, device=x.device))
        y_pred = self(x[:, :-1, :]+noise, t[:, :-1] if t is not None else None,
                      t[:, 1:] if t is not None else None,
                      padding_mask[:, :-1] if padding_mask is not None else None)
        if self.target_normalizer is not None:
            y_target = self.target_normalizer(y_target, padding_mask)
        masked = (torch.arange(L).unsqueeze(0) >= self.mask_n_firsts)
        if padding_mask is not None:
            masked = masked * ~padding_mask
        if weights is not None:
            masked = masked * weights
        return MSE(y_pred, y_target[:, 1:, :], masked[:, 1:].unsqueeze(-1))

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
        X = torch.stack([torch.cat([x, torch.zeros([padded_sequence_length-len(x), len(self.targets)])])
                         for x in Xs if len(x) <= padded_sequence_length], dim=0)
        padding_mask = torch.stack([(torch.arange(padded_sequence_length) >= len(x))
                                    for x in Xs if len(x) <= padded_sequence_length], dim=0)
        if self.time_column is not None:
            Ts = [named_to_tensor(x, [self.time_column])
                for _, x in df.groupby(self.observation_column)]
            T = torch.stack([torch.cat([t, torch.zeros([padded_sequence_length-len(t), 1])])
                             for t in Ts if len(t) <= padded_sequence_length], dim=0)
        else:
            T = None
        if device is not None:
            X = X.to(device)
            if T is not None:
                T = T.to(device)
            padding_mask = padding_mask.to(device)
        return X, T, padding_mask

    def _y_to_tensor(self, df: pd.DataFrame, device: Optional[torch.device] = None,
                     padded_sequence_length: Optional[int] = None) -> torch.Tensor:
        Ys = [named_to_tensor(y, self.targets) for _, y in df.groupby(self.observation_column)]
        if padded_sequence_length is None:
            padded_sequence_length = max(len(y) for y in Ys)
        Y = torch.stack([torch.cat([y, torch.zeros([padded_sequence_length-len(y), len(self.targets)])])
                         for y in Ys if len(y) <= padded_sequence_length], dim=0)
        if device is not None:
            Y = Y.to(device)
        return Y

    def _tensor_to_y(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_dataframe(tensor, self.targets)

    def predict(self, past: pd.DataFrame, future: pd.DataFrame):
        self.eval()
        dfs = []
        for obs in future[self.observation_column].unique():
            df_past = past[past[self.observation_column] == obs]
            df_future = future[future[self.observation_column] == obs]
            df_future = df_future.copy()
            df_future[list(self.targets)] = None
            indexes = [len(self.inputs) + self.targets.index(x) if x in self.targets else i
                    for i, x in enumerate(self.inputs)]
            X_past, T, _ = self._x_to_tensor(df_past, device=self.device)
            X_future, T_future, _ = self._x_to_tensor(df_future, device=self.device)
            X = X_past
            with torch.no_grad():
                for i in range(X_future.shape[1]):
                    if self.input_normalizer is not None:
                        X = self.input_normalizer(X, None)
                    if self.positional_encoding is not None:
                        X = self.positional_encoding(X)
                    if T_future is not None:
                        T = torch.cat([T, T_future[:, i:i+1, :]], dim=1)
                        attention_kwargs = {"query_positions": T[:, :-1, :], "key_positions": T[:, 1:, :]}
                    else:
                        attention_kwargs = {}
                    Y = self.transformer_encoder(self.embedding(X), None, attention_kwargs=attention_kwargs)[:, -1:, :]
                    Y = self.head(Y)
                    if self.target_normalizer is not None:
                        Y = self.target_normalizer.unscale(Y)
                    Y = torch.cat([X_future[:, i:i+1, :], Y], dim=2)[..., indexes]
                    X = torch.cat([X, Y], dim=1)
            df = pd.DataFrame(data=X.squeeze(0).detach().cpu().numpy()[len(df_past):], columns=self.targets)
            df[self.observation_column] = obs
            if T_future is not None:
                df[self.time_column] = T_future.squeeze(0).detach().cpu().numpy().reshape(-1)
            dfs.append(df)
        return pd.concat(dfs)
