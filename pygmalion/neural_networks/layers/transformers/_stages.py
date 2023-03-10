import torch
from typing import Optional
from ._multihead_attention import MultiHeadAttention
from torch.utils.checkpoint import checkpoint


class TransformerEncoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None,
                 activation: str = "relu"):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = getattr(torch, activation)
        self.self_attention = MultiHeadAttention(projection_dim, n_heads)
        self.intermediate_norm = torch.nn.LayerNorm(dim)
        self.intermediate_dropout = torch.nn.Dropout(dropout)
        self.expand = torch.nn.Linear(dim, dim * 4)
        self.contract = torch.nn.Linear(dim * 4, dim)
        self.out_dropout = torch.nn.Dropout(dropout)
        self.out_norm = torch.nn.LayerNorm(dim)

    def forward(self, X, mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None):
        """
        Parameter
        ---------
        X : torch.Tensor
            Tensor of shape (N, L, F) with
            * N sentences count
            * L sequence length
            * F number of features
        mask : torch.Tensor or None
            mask to apply for the attention
        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, F)
        """
        X = X.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        N, L, _ = X.shape
        input = X.reshape(N * L, -1)
        X = self.self_attention(X, X, mask=mask, padding_mask=padding_mask).reshape(N * L, -1)
        X = self.intermediate_dropout(X) + input
        X = self.intermediate_norm(X)
        input = X
        X = self.activation(self.expand(X))
        X = self.out_dropout(self.contract(X)) + input
        X = self.out_norm(X)
        return X.reshape(N, L, -1)

    @property
    def device(self) -> torch.device:
        return self.self_attention.key.weight.device


class TransformerDecoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = getattr(torch, activation)
        self.masked_attention = MultiHeadAttention(projection_dim, n_heads)
        self.first_dropout = torch.nn.Dropout(dropout)
        self.first_norm = torch.nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(projection_dim, n_heads)
        self.second_dropout = torch.nn.Dropout(dropout)
        self.second_norm = torch.nn.LayerNorm(dim)
        self.expand = torch.nn.Linear(dim, 4 * dim)
        self.contract = torch.nn.Linear(4 * dim, dim)
        self.out_dropout = torch.nn.Dropout(dropout)
        self.out_norm = torch.nn.LayerNorm(dim)

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None):
        """
        Parameter
        ---------
        encoded : torch.Tensor
            Tensor of shape (N, L, F)
        Y : torch.Tensor
            Tensor of shape (N, L, F)
        mask : torch.Tensor or None
            mask to apply for the attention
        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, F)
        """
        encoded = encoded.to(self.device)
        Y = Y.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        N, L, _ = Y.shape
        input = Y.reshape(N * L, -1)
        Y = self.masked_attention(Y, Y, mask=mask, padding_mask=padding_mask).reshape(N * L, -1)
        Y = self.first_norm(self.first_dropout(Y) + input).reshape(N, L, -1)
        input = Y.reshape(N * L, -1)
        Y = self.attention(Y, encoded, mask=None, padding_mask=padding_mask).reshape(N * L, -1)
        Y = self.second_norm(self.second_dropout(Y) + input)
        input = Y
        Y = self.out_dropout(self.contract(self.activation(self.expand(Y))))
        Y = self.out_norm(Y + input)
        return Y.reshape(N, L, -1)

    @property
    def device(self) -> torch.device:
        return self.masked_attention.key.weight.device


class TransformerEncoder(torch.nn.Module):

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 low_memory: bool = True):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        self.low_memory = low_memory
        for stage in range(n_stages):
            self.stages.append(TransformerEncoderStage(projection_dim, n_heads,
                                                       dropout=dropout, activation=activation))

    def forward(self, X, mask=None, padding_mask: Optional[torch.Tensor] = None):
        for stage in self.stages:
            if self.low_memory and self.training:
                X = checkpoint(stage, X, mask, padding_mask)
            else:
                X = stage(X, mask=mask, padding_mask=padding_mask)
        return X


class TransformerDecoder(torch.nn.Module):

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 low_memory: bool = True):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        self.low_memory = low_memory
        for stage in range(n_stages):
            self.stages.append(TransformerDecoderStage(projection_dim, n_heads,
                                                       dropout=dropout, activation=activation))

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None):
        for stage in self.stages:
            if self.low_memory and self.training:
                Y = checkpoint(stage, encoded, Y, mask, padding_mask)
            else:
                Y = stage(encoded, Y, mask=mask, padding_mask=padding_mask)
        return Y
