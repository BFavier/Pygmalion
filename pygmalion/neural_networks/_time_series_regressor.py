import torch
import pandas as pd
from typing import Optional, Iterable, Union
from .layers.transformers import TransformerEncoder, TransformerDecoder, ATTENTION_TYPE, ScaledDotProductAttention
from .layers import Normalizer
from ._conversions import named_to_tensor, tensor_to_dataframe, floats_to_tensor
from ._neural_network import NeuralNetwork
from ._loss_functions import MSE


class TimeSeriesRegressor(NeuralNetwork):

    def __init__(self, inputs: Iterable[str], targets: Iterable[str],
                 observation_column: str, time_column: Optional[str],
                 n_stages: int, projection_dim: int, n_heads: int,
                 activation: str = "relu",
                 dropout: Optional[float] = None,
                 normalize: bool = True,
                 gradient_checkpointing: bool = True,
                 positional_encoding: bool = True,
                 attention_type: ATTENTION_TYPE = ScaledDotProductAttention,
                 send_time_to_attention: bool = False,
                 mask_future: bool = False,
                 attention_kwargs: dict = {}):
        """
        Parameters
        ----------
        ...
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
        gradient_checkpointing : bool
            If True, uses gradient checkpointing to reduce memory usage during
            training at the expense of computation time.
        positional_encoding : bool
            If True, absolute time is encoded by adding linear projection of time to embeddings
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
        self.send_time_to_attention = send_time_to_attention
        embedding_dim = projection_dim*n_heads
        self.input_normalizer = Normalizer(-1, len(inputs)) if normalize else None
        self.target_normalizer = Normalizer(-1, len(targets)) if normalize else None
        self.time_normalizer = Normalizer(-1, 1) if normalize else None
        self.inputs_embedding = torch.nn.Linear(len(inputs), embedding_dim)
        self.targets_embedding = torch.nn.parameter.Parameter(torch.zeros((1, 1, embedding_dim)))
        if positional_encoding:
            self.positional_encoding = torch.nn.Linear(1, embedding_dim)
        else:
            self.positional_encoding = None
        self.encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout, activation=activation,
                                          attention_type=attention_type,
                                          gradient_checkpointing=gradient_checkpointing,
                                          mask_future=mask_future,
                                          **attention_kwargs)
        self.decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout, activation=activation,
                                          attention_type=attention_type,
                                          gradient_checkpointing=gradient_checkpointing,
                                          mask_future=False,
                                          self_attention=False,
                                          **attention_kwargs)
        self.head = torch.nn.Linear(embedding_dim, len(targets))

    def forward(self, X: torch.Tensor, Tx: Optional[torch.Tensor],
                x_padding_mask: Optional[torch.Tensor],
                Ly: int, Ty: Optional[torch.Tensor],
                y_padding_mask: Optional[torch.Tensor]):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            inputs of the model at each time step.
            tensor of floats of shape (N, Lx, D)
        Tx : torch.Tensor or None
            Time associated to inputs values points in the X sequences.
            tensor of floats of shape (N, Lx)
        x_padding_mask : torch.Tensor or None
            Whether each point in the inputs sequences are padding.
            tensor of booleans of shape (N, Lx)
        Ly : int
            length of the target sequence
        Ty : torch.Tensor or None
            Time associated to target values points in the Y sequences.
            tensor of floats of shape (N, Ly)
        y_padding_mask : torch.Tensor or None
            Whether each point in the targets sequences are padding.
            tensor of booleans of shape (N, Ly)

        Returns
        -------
        torch.Tensor :
            Predicted target Y at the time steps of inputs X
            tensor of floats of shape (N, Lx, n_targets)
        """
        X = X.to(self.device)
        N, _, _ = X.shape
        if x_padding_mask is not None:
            x_padding_mask = x_padding_mask.to(self.device)
        if y_padding_mask is not None:
            y_padding_mask = y_padding_mask.to(self.device)
        if Tx is not None:
            Tx = Tx.to(dtype=X.dtype, device=X.device)
        else:
            Tx = torch.arange(0, x_padding_mask.shape[1], dtype=X.dtype, device=self.device).reshape(1, -1, 1).expand(N, -1, -1)
        if Ty is not None:
            Ty = Ty.to(dtype=X.dtype, device=X.device)
        else:
            Ty = (torch.arange(0, Ly, dtype=X.dtype, device=self.device).reshape(1, -1, 1).expand(N, -1, -1) + (~x_padding_mask).sum(dim=-1).reshape(-1, 1, 1))
        if self.input_normalizer is not None:
            X = self.input_normalizer(X, x_padding_mask)
        if self.time_normalizer is not None:
            Tx = self.time_normalizer(Tx, x_padding_mask)
            Ty = self.time_normalizer(Ty, y_padding_mask)
        X = self.inputs_embedding(X)
        Y = self.targets_embedding.expand(len(X), Ly, -1)
        if self.positional_encoding is not None:
            X = X + self.positional_encoding(Tx)
            Y = Y + self.positional_encoding(Ty)
        encoded = self.encoder(X, x_padding_mask,
                               attention_kwargs={"query_positions": Tx, "key_positions": Tx} if self.send_time_to_attention else {})
        decoded = self.decoder(Y, encoded, y_padding_mask, x_padding_mask,
                               self_attention_kwargs={"query_positions": Tx, "key_positions": Tx} if self.send_time_to_attention else {},
                               cross_attention_kwargs={"query_positions": Tx, "key_positions": Ty} if self.send_time_to_attention else {})
        return self.head(decoded)

    def loss(self, X: torch.Tensor, Tx: Optional[torch.Tensor], x_padding_mask: torch.Tensor,
             Y: torch.Tensor, Ty: Optional[torch.Tensor], y_padding_mask: torch.Tensor,
             weights: Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (N, L_inputs, n_inputs)
        Tx : torch.Tensor or None
            if provided, the time at each point of the sequence
            tensor of floats of shape (N, L_inputs)
        x_padding_mask : torch.Tensor
            tensor of booleans of shape (N, L_inputs)
        Y : torch.Tensor
            tensor of floats of shape (N, L_targets, n_targets)
        Ty : torch.Tensor or None
            if provided, the time at each point of the sequence
            tensor of floats of shape (N, L_targets)
        y_padding_mask : torch.Tensor
            tensor of booleans of shape (N, L_targets)
        weights : torch.Tensor or None
            tensor of weights of shape (N, L_targets) or None.
        """
        y_pred = self(X, Tx, x_padding_mask, Y.shape[1], Ty, y_padding_mask)
        w = (weights * ~y_padding_mask).unsqueeze(-1) if weights is not None else ~y_padding_mask.unsqueeze(-1)
        if self.target_normalizer is not None:
            Y = self.target_normalizer(Y, y_padding_mask)
        return MSE(y_pred, Y, w)

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def data_to_tensor(self, inputs: pd.DataFrame, targets: pd.DataFrame,
                       device: Optional[torch.device] = None,
                       input_sequence_length: Optional[int] = None,
                       target_sequence_length: Optional[int] = None,
                       raise_on_longer_sequences: bool = False) -> tuple:
        diff = set.difference(set(inputs[self.observation_column]), set(targets[self.observation_column]))
        if len(diff) > 0:
            raise ValueError(f"Some values of the '{self.observation_column}' column are present in only one of the input/target dataframes: {diff}")
        X, Tx, x_padding_mask = self._x_to_tensor(inputs, device, input_sequence_length, raise_on_longer_sequences)
        Y, Ty, y_padding_mask = self._y_to_tensor(targets, device, target_sequence_length)
        return X, Tx, x_padding_mask, Y, Ty, y_padding_mask

    def _x_to_tensor(self, df: pd.DataFrame, device: Optional[torch.device] = None,
                     padded_sequence_length: Optional[int] = None,
                     raise_on_longer_sequences: bool = False):
        df = df.sort_values(by=[self.observation_column]+([self.time_column] if self.time_column is not None else []))
        if raise_on_longer_sequences and padded_sequence_length is not None:
            for obs, x in df.groupby(self.observation_column):
                if len(x) > padded_sequence_length:
                    raise RuntimeError(f"Found sequence longer than {padded_sequence_length} for observation '{obs}'")
        Xs = [named_to_tensor(x, self.inputs) for _, x in df.groupby(self.observation_column)]
        if padded_sequence_length is None:
            padded_sequence_length = max(len(x) for x in Xs)
        X = torch.stack([torch.cat([x, torch.zeros([padded_sequence_length-len(x), len(self.inputs)])])
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
                     padded_sequence_length: Optional[int] = None,
                     raise_on_longer_sequences: bool = False):
        df = df.sort_values(by=[self.observation_column]+([self.time_column] if self.time_column is not None else []))
        if raise_on_longer_sequences and padded_sequence_length is not None:
            for obs, y in df.groupby(self.observation_column):
                if len(y) > padded_sequence_length:
                    raise RuntimeError(f"Found sequence longer than {padded_sequence_length} for observation '{obs}'")
        Ys = [named_to_tensor(y, self.targets) for _, y in df.groupby(self.observation_column)]
        if padded_sequence_length is None:
            padded_sequence_length = max(len(y) for y in Ys)
        Y = torch.stack([torch.cat([y, torch.zeros([padded_sequence_length-len(y), len(self.targets)])])
                         for y in Ys if len(y) <= padded_sequence_length], dim=0)
        padding_mask = torch.stack([(torch.arange(padded_sequence_length) >= len(y))
                                    for y in Ys if len(y) <= padded_sequence_length], dim=0)
        if self.time_column is not None:
            Ts = [named_to_tensor(y, [self.time_column])
                  for _, y in df.groupby(self.observation_column)]
            T = torch.stack([torch.cat([t, torch.zeros([padded_sequence_length-len(t), 1])])
                             for t in Ts if len(t) <= padded_sequence_length], dim=0)
        else:
            T = None
        if device is not None:
            Y = Y.to(device)
            if T is not None:
                T = T.to(device)
            padding_mask = padding_mask.to(device)
        return Y, T, padding_mask

    def _tensor_to_y(self, y_pred: torch.Tensor, padding_mask: torch.Tensor) -> pd.DataFrame:
       N, L, D = y_pred.shape
       df = tensor_to_dataframe(y_pred.reshape(-1, D), self.targets)
       return df[~padding_mask.reshape(-1).cpu().numpy()]

    def predict(self, df: pd.DataFrame, times: Union[Iterable[float], int]=None) -> pd.DataFrame:
        """
        """
        self.eval()
        df = df.sort_values(by=[self.observation_column]+([self.time_column] if self.time_column is not None else []))
        X, Tx, x_padding_mask = self._x_to_tensor(df, device=self.device)
        Ly = times if isinstance(times, int) else len(times)
        Ty = None if isinstance(times, int) else floats_to_tensor(times, device=self.device).reshape(1, -1, 1)
        with torch.no_grad():
            y_pred = self(X, Tx, x_padding_mask,
                          Ly, Ty, y_padding_mask=None)
        if self.target_normalizer is not None:
            y_pred = self.target_normalizer.unscale(y_pred)
        y_pred = y_pred.cpu().numpy()
        dfs = [pd.DataFrame(data=data, columns=self.targets) for data in y_pred]
        for sub, obs in zip(dfs, df[self.observation_column].unique()):
            sub[self.observation_column] = obs
        return pd.concat(dfs)
