import torch
from typing import Optional
from ._multihead_attention import MultiHeadAttention, ATTENTION_TYPE
from torch.utils.checkpoint import checkpoint


class TransformerEncoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None,
                 activation: str = "relu", RPE_radius: Optional[int] = None,
                 attention_type: ATTENTION_TYPE = "scaled dot product",
                 **kwargs):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = getattr(torch, activation)
        self.self_attention = MultiHeadAttention(projection_dim, n_heads, False,
                                                 RPE_radius=RPE_radius,
                                                 attention_type=attention_type,
                                                 **kwargs)
        self.intermediate_norm = torch.nn.LayerNorm(dim)
        self.intermediate_dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.expand = torch.nn.Linear(dim, dim * 4)
        self.contract = torch.nn.Linear(dim * 4, dim)
        self.out_dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.out_norm = torch.nn.LayerNorm(dim)

    def forward(self, X: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        """
        Parameter
        ---------
        X : torch.Tensor
            Tensor of shape (N, L, D) with
            * N sentences count
            * L sequence length
            * D number of features
        padding_mask : torch.tensor or None
            tensor of booleans of shape (N, L) to ignore
        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, D)
        """
        X = X.to(self.device)
        N, L, _ = X.shape
        input = X.reshape(N * L, -1)
        X = self.self_attention(X, X, padding_mask, padding_mask).reshape(N * L, -1)
        if self.intermediate_dropout is not None:
            X = self.intermediate_dropout(X) + input
        X = self.intermediate_norm(X)
        input = X
        X = self.contract(self.activation(self.expand(X)))
        if self.out_dropout is not None:
            X = self.out_dropout(X)
        X = self.out_norm(X + input)
        return X.reshape(N, L, -1)

    @property
    def device(self) -> torch.device:
        return self.self_attention.key.weight.device


class TransformerDecoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 RPE_radius: Optional[int] = None,
                 attention_type: ATTENTION_TYPE = "scaled dot product",
                 **kwargs):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = getattr(torch, activation)
        self.masked_attention = MultiHeadAttention(projection_dim, n_heads, True,
                                                   RPE_radius=RPE_radius,
                                                   attention_type=attention_type,
                                                   **kwargs)
        self.first_dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.first_norm = torch.nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(projection_dim, n_heads, False,
                                            RPE_radius=RPE_radius,
                                            attention_type=attention_type)
        self.second_dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.second_norm = torch.nn.LayerNorm(dim)
        self.expand = torch.nn.Linear(dim, 4 * dim)
        self.contract = torch.nn.Linear(4 * dim, dim)
        self.out_dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.out_norm = torch.nn.LayerNorm(dim)

    def forward(self, Y, encoded,
                encoded_padding_mask: Optional[torch.Tensor] = None):
        """
        Parameter
        ---------
        encoded : torch.Tensor
            Tensor of shape (N, Lk, F)
        Y : torch.Tensor
            Tensor of shape (N, Lq, F)
        encoded_padding_mask : torch.Tensor or None
            mask of shape (N, Lk)
        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, F)
        """
        encoded = encoded.to(self.device)
        Y = Y.to(self.device)
        N, L, _ = Y.shape
        input = Y.reshape(N * L, -1)
        Y = self.masked_attention(Y, Y).reshape(N * L, -1)
        if self.first_dropout is not None:
            Y = self.first_dropout(Y)
        Y = self.first_norm(Y + input).reshape(N, L, -1)
        input = Y.reshape(N * L, -1)
        Y = self.attention(Y, encoded, query_padding_mask=None,
                           key_padding_mask=encoded_padding_mask).reshape(N * L, -1)
        if self.second_dropout is not None:
            Y = self.second_dropout(Y)
        Y = self.second_norm(Y + input)
        input = Y
        Y = self.contract(self.activation(self.expand(Y)))
        if self.out_dropout is not None:
            Y = self.out_dropout(Y)
        Y = self.out_norm(Y + input)
        return Y.reshape(N, L, -1)

    @property
    def device(self) -> torch.device:
        return self.masked_attention.key.weight.device
