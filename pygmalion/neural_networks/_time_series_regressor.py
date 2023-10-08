import torch
import pandas as pd
from warnings import warn
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
            if True, the inputs and targets ar enormalized
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
        self.input_normalizer = Normalizer(-1, len(inputs)) if normalize else None
        self.target_normalizer = Normalizer(-1, len(targets)) if normalize else None
        self.time_normalizer = Normalizer(-1, 1) if normalize and time_column is not None else None
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
            tensor of floats of shape (N, L, D) of properties at current time step
        T : torch.Tensor or None
            tensor of floats of shape (N, L) of time at current time step
        T_next : torch.Tensor or None
            tensor of floats of shape (N, L) of time at nexst time step
        padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, L) of whether the current time step is padding

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
        if self.time_normalizer is not None:
            T = self.time_normalizer(T)
        X = self.embedding(X)
        N, L, _ = X.shape
        if self.positional_encoding is not None:
            X = self.positional_encoding(X)
        X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        attention_kwargs = {"query_positions": T_next, "key_positions": T} if T is not None else {}
        X = self.transformer_encoder(X, padding_mask, attention_kwargs=attention_kwargs)
        return self.head(X)

    def _predict_tensor(self, X: torch.Tensor, T: Optional[torch.Tensor]) -> torch.Tensor:
        """
        """
        N, L, _ = X.shape
        input_indexes = [len(self.inputs) + self.targets.index(x) if x in self.targets else i
                         for i, x in enumerate(self.inputs)]  # index of the newly predicted model inputs in the concatenation of input and prediction
        X = X[:, :1, :]
        with torch.no_grad():
            for i in range(L-1):
                Y = self(X, T[:, :i+1] if T is not None else None,
                         T[:, 1:i+2] if T is not None else None, None)[:, -1:, :]
                if self.target_normalizer is not None:
                    Y = self.target_normalizer.unscale(Y)
                # adding predicted difference to previous time step values
                referential = torch.cat([X[:, -1:, :], torch.zeros((N, 1, 1), device=self.device)], dim=-1)  # value of the target at previous time step, if the target is in the inputs, otherwise 0
                index = [self.inputs.index(c) if c in self.inputs else -1 for c in self.targets]
                Y = Y + referential[..., index]
                # append to inputs
                new_inputs = torch.cat([X[:, -1:, :], Y], dim=2)[..., input_indexes]
                X = torch.cat([X, new_inputs], dim=1)
        return X

    def loss(self, X: torch.Tensor, T: Optional[torch.Tensor], padding_mask: Optional[torch.Tensor],
             Y_target: torch.Tensor, weights: Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of floats of shape (N, L, D) of inputs at each time step
        t : torch.Tensor or None
            if provided, the time as a tensor of floats of shape (N, L)
        padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, L)
        y_target : torch.Tensor
            tensor of floats of shape (N, L, D) of targets at each time step
        """
        N, L, D = X.shape
        X, Y_target = X.to(self.device), Y_target.to(self.device)
        if T is not None:
            T = T.to(self.device)
        if self.training:
            inputs = self._predict_tensor(X, T)
            weights = None
        else:
            inputs = X
            weights = None
        y_pred = self(inputs[:, :-1, :],
                      T[:, :-1] if T is not None else None,
                      T[:, 1:] if T is not None else None,
                      padding_mask[:, :-1] if padding_mask is not None else None)
        referential = torch.cat([X[:, :-1, :], torch.zeros((N, L-1, 1), device=self.device)], dim=-1)  # value of the target at previous time step, if the target is in the inputs, otherwise 0
        index = [self.inputs.index(c) if c in self.inputs else -1 for c in self.targets]
        target = Y_target[:, 1:, :] - referential[..., index]
        if self.target_normalizer is not None:
            target = self.target_normalizer(target, padding_mask[:, :-1] if padding_mask is not None else None)
        masked = (torch.arange(L, device=self.device).unsqueeze(0) >= self.n_min_points)
        if padding_mask is not None:
            masked = masked * ~padding_mask.to(self.device)
        if weights is not None:
            masked = masked * weights
        return MSE(y_pred, target, masked[:, 1:].unsqueeze(-1))

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
            if len(df_past) < self.n_min_points:
                warn(f"Tried predicting time series from an initial condition of less than n_min_points={self.n_min_points} points for observation {obs}")
            df_future = future[future[self.observation_column] == obs]
            df_future = df_future.copy()
            df_future[list(self.targets)] = None
            input_indexes = [len(self.inputs) + self.targets.index(x) if x in self.targets else i
                             for i, x in enumerate(self.inputs)]  # index of the newly predicted model inputs in the concatenation of input and prediction
            X_past, T, _ = self._x_to_tensor(df_past, device=self.device)
            X_future, T_future, _ = self._x_to_tensor(df_future, device=self.device)
            X = X_past
            predictions = torch.zeros((0, len(self.targets)), device=self.device, dtype=torch.float)
            with torch.no_grad():
                for i in range(X_future.shape[1]):
                    if T_future is not None:
                        T = torch.cat([T, T_future[:, i:i+1, :]], dim=1)
                    Y = self(X, T[:, :-1] if T is not None else None,
                             T[:, 1:] if T is not None else None, None)[:, -1:, :]
                    if self.target_normalizer is not None:
                        Y = self.target_normalizer.unscale(Y)
                    # adding predicted difference to previous time step values
                    referential = torch.cat([X[:, -1:, :], torch.zeros((1, 1, 1), device=self.device)], dim=-1)  # value of the target at previous time step, if the target is in the inputs, otherwise 0
                    index = [self.inputs.index(c) if c in self.inputs else -1 for c in self.targets]
                    Y = Y + referential[..., index]
                    # append to predictions and inputs
                    predictions = torch.cat([predictions, Y.squeeze(0)], dim=0)
                    new_inputs = torch.cat([X_future[:, i:i+1, :], Y], dim=2)[..., input_indexes]
                    X = torch.cat([X, new_inputs], dim=1)
            df = pd.DataFrame(data=predictions.detach().cpu().numpy(), columns=self.targets)
            df[self.observation_column] = obs
            if T_future is not None:
                df[self.time_column] = T_future.squeeze(0).detach().cpu().numpy().reshape(-1)
            dfs.append(df)
        return pd.concat(dfs)
