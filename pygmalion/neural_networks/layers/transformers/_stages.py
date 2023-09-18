import torch
from typing import Optional
from .multihead_attention import ATTENTION_TYPE, ScaledDotProductAttention
from pygmalion.neural_networks.layers._dropout import Dropout
from pygmalion.neural_networks.layers._activation import Activation


class TransformerEncoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None,
                 activation: str = "relu",
                 attention_type: ATTENTION_TYPE = ScaledDotProductAttention,
                 mask_future: bool = False,
                 **kwargs):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = Activation(activation)
        self.self_attention = attention_type(projection_dim, n_heads, mask_future=mask_future, **kwargs)
        self.intermediate_norm = torch.nn.LayerNorm(dim)
        self.intermediate_dropout = Dropout(dropout)
        self.expand = torch.nn.Linear(dim, dim * 4)
        self.contract = torch.nn.Linear(dim * 4, dim)
        self.out_dropout = Dropout(dropout)
        self.out_norm = torch.nn.LayerNorm(dim)

    def forward(self, X: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                attention_kwargs: dict = {}):
        """
        Parameter
        ---------
        X : torch.Tensor
            Tensor of shape (N, L, D) with
            * N sentences count
            * L sequence length
            * D number of features
        padding_mask : torch.tensor or None
            tensor of booleans of shape (N, L) of tokens to ignore
        attention_kwargs : dict
            kwargs passed to self_attention

        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, D)
        """
        X = X.to(self.device)
        N, L, _ = X.shape
        input = X.reshape(N * L, -1)
        X = self.self_attention(X, X, padding_mask, padding_mask, **attention_kwargs).reshape(N * L, -1)
        X = self.intermediate_dropout(X) + input
        X = self.intermediate_norm(X)
        input = X
        X = self.contract(self.activation(self.expand(X)))
        X = self.out_dropout(X)
        X = self.out_norm(X + input)
        return X.reshape(N, L, -1)
    
    def generate(self, X: torch.Tensor):
        pass

    @property
    def device(self) -> torch.device:
        return self.self_attention.key.weight.device


class TransformerDecoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 attention_type: ATTENTION_TYPE = ScaledDotProductAttention,
                 **kwargs):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = Activation(activation)
        self.masked_self_attention = attention_type(projection_dim, n_heads, mask_future=True, **kwargs)
        self.first_dropout = Dropout(dropout)
        self.first_norm = torch.nn.LayerNorm(dim)
        self.attention = attention_type(projection_dim, n_heads, mask_future=False, **kwargs)
        self.second_dropout = Dropout(dropout)
        self.second_norm = torch.nn.LayerNorm(dim)
        self.expand = torch.nn.Linear(dim, 4 * dim)
        self.contract = torch.nn.Linear(4 * dim, dim)
        self.out_dropout = Dropout(dropout)
        self.out_norm = torch.nn.LayerNorm(dim)

    def forward(self, Y, encoded,
                encoded_padding_mask: Optional[torch.Tensor] = None):
        """
        Parameter
        ---------
        Y : torch.Tensor
            Tensor of shape (N, Lq, D)
        encoded : torch.Tensor
            Tensor of shape (N, Lk, D)
        encoded_padding_mask : torch.Tensor or None
            mask of shape (N, Lk)

        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, D)
        """
        encoded = encoded.to(self.device)
        Y = Y.to(self.device)
        N, L, _ = Y.shape
        input = Y.reshape(N * L, -1)
        Y = self.masked_self_attention(Y, Y).reshape(N * L, -1)
        Y = self.first_dropout(Y)
        Y = self.first_norm(Y + input).reshape(N, L, -1)
        input = Y.reshape(N * L, -1)
        Y = self.attention(Y, encoded, query_mask=None,
                           key_mask=encoded_padding_mask).reshape(N * L, -1)
        Y = self.second_dropout(Y)
        Y = self.second_norm(Y + input)
        input = Y
        Y = self.contract(self.activation(self.expand(Y)))
        Y = self.out_dropout(Y)
        Y = self.out_norm(Y + input)
        return Y.reshape(N, L, -1)

    def predict(self, Y, encoded,
                encoded_padding_mask: Optional[torch.Tensor]):
        """
        Efficiently predict the next representation
        of the last token in the Y sequence

        Parameter
        ---------
        Y : torch.Tensor
            Tensor of shape (N, Lq, D)
        encoded : torch.Tensor
            Tensor of shape (N, Lk, D)
        encoded_padding_mask : torch.Tensor or None
            mask of shape (N, Lk)

        Returns
        -------
        torch.Tensor
            tensor of shape (N, 1, D)
        """
        assert not self.training
        encoded = encoded.to(self.device)
        Y = Y.to(self.device)
        N, L, _ = Y.shape
        Q = Y[:, -1:, :]
        input = Q.reshape(N, -1)
        Q = self.masked_self_attention(Q, Y, future_offset=L-1).reshape(N, -1)
        Q = self.first_norm(Q + input).reshape(N, 1, -1)
        input = Q.reshape(N, -1)
        Q = self.attention(Q, encoded, query_mask=None,
                           key_mask=encoded_padding_mask,
                           future_offset=L-1).reshape(N, -1)
        Q = self.second_norm(Q + input)
        input = Q
        Q = self.contract(self.activation(self.expand(Q)))
        Q = self.out_norm(Q + input)
        return Q.reshape(N, 1, -1)

    @property
    def device(self) -> torch.device:
        return self.masked_self_attention.key.weight.device
