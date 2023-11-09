import torch
import pandas as pd
from warnings import warn
from typing import Union, Optional, Iterable
from .layers.transformers import TransformerEncoder, TransformerDecoder, ATTENTION_TYPE, FourrierKernelAttention
from .layers.positional_encoding import POSITIONAL_ENCODING_TYPE
from .layers import Dropout, Normalizer, Dense
from ._conversions import named_to_tensor, tensor_to_dataframe
from ._neural_network import NeuralNetwork
from ._loss_functions import MSE


class TimeSeriesRegressor(NeuralNetwork):

    def __init__(self, inputs: Iterable[str], targets: Iterable[str],
                 observation_column: str, time_column: Optional[str],
                 n_stages: int, projection_dim: int, n_heads: int,
                 activation: str = "relu",
                 dropout: Optional[float] = None,
                 normalize: bool = True,
                 n_min_points: int = 1,
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
            if True, the inputs and targets are normalized
        n_min_points : int
            Minimum number of points as initial condition to be able to make a prediction.
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
        self.n_min_points = n_min_points
        embedding_dim = projection_dim*n_heads
        self.input_normalizer = Normalizer(-1, max(1, len(inputs))) if normalize else None
        self.target_normalizer = Normalizer(-1, len(targets)) if normalize else None
        self.time_normalizer = Normalizer(-1, 1) if normalize and time_column is not None else None
        self.inputs_embedding = torch.nn.Linear(max(1, len(inputs)), embedding_dim)
        self.targets_embedding = torch.nn.Linear(len(targets), embedding_dim)
        if positional_encoding_type is None:
            self.positional_encoding = None
        else:
            self.positional_encoding = positional_encoding_type(embedding_dim, **positional_encoding_kwargs)
        self.encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout, activation=activation,
                                          attention_type=attention_type,
                                          gradient_checkpointing=gradient_checkpointing,
                                          mask_future=True,
                                          **attention_kwargs)
        self.decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout, activation=activation,
                                          attention_type=attention_type,
                                          gradient_checkpointing=gradient_checkpointing,
                                          mask_future=True,
                                          **attention_kwargs)
        self.head = torch.nn.Linear(embedding_dim, len(targets))

    def forward(self, X: torch.Tensor, Y: torch.Tensor,
                Tx: Optional[torch.Tensor], Ty: Optional[torch.Tensor],
                x_padding_mask: Optional[torch.Tensor], y_padding_mask: Optional[torch.Tensor]):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            inputs of the model at each time step.
            tensor of floats of shape (N, Lx, D)
        Y : torch.Tensor
            observed values of the model's targets at each time step.
            tensor of floats of shape (N, Ly, D)
        Tx : torch.Tensor or None
            Time associated to inputs values points in the X sequences.
            tensor of floats of shape (N, Lx)
        Ty : torch.Tensor or None
            Time associated to target values points in the Y sequences.
            tensor of floats of shape (N, Ly)
        x_padding_mask : torch.Tensor or None
            Whether each point in the inputs sequences are padding.
            tensor of booleans of shape (N, Lx)
        y_padding_mask : torch.Tensor or None
            Whether each point in the targets sequences are padding.
            tensor of booleans of shape (N, Lx)

        Returns
        -------
        torch.Tensor :
            Predicted target Y at the time steps of inputs X
            tensor of floats of shape (N, Lx, n_targets)
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        x_padding_mask = x_padding_mask.to(self.device)
        y_padding_mask = y_padding_mask.to(self.device)
        if Tx is not None:
            Tx = Tx.to(self.device)
        if Ty is not None:
            Ty = Ty.to(self.device)
        if self.input_normalizer is not None:
            X = self.input_normalizer(X, x_padding_mask)
        if self.time_normalizer is not None:
            Tx = self.time_normalizer(Tx, x_padding_mask)
            Ty = self.time_normalizer(Ty, y_padding_mask)
        X = self.inputs_embedding(X)
        Y = self.targets_embedding(Y)
        if self.positional_encoding is not None:
            X = self.positional_encoding(X)
            Y = self.positional_encoding(Y)
        encoded = self.encoder(Y, y_padding_mask,
                               attention_kwargs={"query_positions": Ty, "key_positions": Ty} if Ty is not None else {})
        decoded = self.decoder(X, encoded, x_padding_mask, y_padding_mask,
                               self_attention_kwargs={"query_positions": Tx, "key_positions": Tx} if Tx is not None else {},
                               cross_attention_kwargs={"query_positions": Tx, "key_positions": Ty} if Tx is not None else {})
        return self.head(decoded)

    def loss(self, X: torch.Tensor, Y: torch.Tensor, T: Optional[torch.Tensor],
             padding_mask: torch.Tensor, weights: Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (N, L, n_inputs)
        Y : torch.Tensor
            tensor of floats of shape (N, L, n_targets)
        T : torch.Tensor or None
            if provided, the time at each point of the sequence
            tensor of floats of shape (N, L)
        padding_mask : torch.Tensor
            tensor of booleans of shape (N, L)
        y_target : torch.Tensor
            tensor of floats of shape (N, L, D) of targets at each time step
        """
        N, L = padding_mask.shape
        if self.target_normalizer is not None:
            Y = self.target_normalizer(Y, padding_mask)
        random_time_horizon = (torch.arange(L).unsqueeze(0) < (
            self.n_min_points + torch.rand(size=(N,)) * torch.clip(torch.sum(~padding_mask, dim=-1) - self.n_min_points, min=0)
            ).long().unsqueeze(-1))
        y_padding_mask = padding_mask & random_time_horizon
        y_pred = self(X, Y, T, T, padding_mask, y_padding_mask)
        return MSE(y_pred, Y, (weights * ~padding_mask).unsqueeze(-1) if weights is not None else ~padding_mask.unsqueeze(-1))

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def data_to_tensor(self, df: pd.DataFrame,
                       device: Optional[torch.device] = None,
                       padded_sequence_length: Optional[int] = None,
                       raise_on_longer_sequences: bool = False) -> tuple:
        X, T, padding_mask = self._x_to_tensor(df, device, padded_sequence_length, raise_on_longer_sequences)
        Y = self._y_to_tensor(df, device, padded_sequence_length)
        return X, Y, T, padding_mask

    def _x_to_tensor(self, df: pd.DataFrame, device: Optional[torch.device] = None,
                     padded_sequence_length: Optional[int] = None,
                     raise_on_longer_sequences: bool = False):
        if raise_on_longer_sequences and padded_sequence_length is not None:
            for obs, x in df.groupby(self.observation_column):
                if len(x) > padded_sequence_length:
                    raise RuntimeError(f"Found sequence longer than {padded_sequence_length} for observation '{obs}'")
        Xs = [named_to_tensor(x, self.inputs) if len(self.inputs) > 0 else torch.ones((len(x), 1)) for _, x in df.groupby(self.observation_column)]
        if padded_sequence_length is None:
            padded_sequence_length = max(len(x) for x in Xs)
        X = torch.stack([torch.cat([x, torch.zeros([padded_sequence_length-len(x), max(1, len(self.inputs))])])
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

    def _tensor_to_y(self, y_pred: torch.Tensor, padding_mask: torch.Tensor) -> pd.DataFrame:
       N, L, D = y_pred.shape
       df = tensor_to_dataframe(y_pred.reshape(-1, D), self.targets)
       return df[~padding_mask.reshape(-1).cpu().numpy()]

    def predict(self, past: pd.DataFrame, future: pd.DataFrame):
        """
        """
        self.eval()
        _, Y, Ty, y_padding_mask = self.data_to_tensor(past, device=self.device)
        if self.target_normalizer is not None:
            Y = self.target_normalizer(Y, y_padding_mask)
        df = pd.concat([past, future]).sort_values(self.observation_column if self.time_column is None else [self.observation_column, self.time_column])
        X, Tx, x_padding_mask = self._x_to_tensor(df, device=self.device)
        with torch.no_grad():
            predicted = self(X, Y, Tx, Ty, x_padding_mask, y_padding_mask)
            if self.target_normalizer is not None:
                predicted = self.target_normalizer.unscale(predicted)
        pred = self._tensor_to_y(predicted, x_padding_mask)
        pred[self.observation_column] = df[self.observation_column]
        if self.time_column is not None:
            pred[self.time_column] = df[self.time_column]
        return pred
