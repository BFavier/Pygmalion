import torch
from typing import Optional
from ._activation import Activation
from ._multi_head_attention import MultiHeadAttention as MHA


class TransformerEncoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = Activation(activation)
        self.self_attention = MHA(projection_dim, n_heads)
        self.intermediate_norm = torch.nn.BatchNorm1d(dim)
        self.intermediate_dropout = torch.nn.Dropout(dropout)
        self.expand = torch.nn.Linear(dim, dim*2)
        self.contract = torch.nn.Linear(dim*2, dim)
        self.out_dropout = torch.nn.Dropout(dropout)
        self.out_norm = torch.nn.BatchNorm1d(dim)

    def forward(self, X, mask: Optional[torch.Tensor] = None):
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
        N, L, _ = X.shape
        input = X.reshape(N*L, -1)
        X = self.self_attention(X, X, mask=mask).reshape(N*L, -1)
        X = self.intermediate_norm(self.intermediate_dropout(X) + input)
        input = X
        X = self.activation(self.expand(X))
        X = self.out_dropout(self.contract(X))
        X = self.out_norm(X + input)
        return X.view(N, L, -1)


class TransformerDecoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = Activation(activation)
        self.masked_attention = MHA(projection_dim, n_heads)
        self.first_dropout = torch.nn.Dropout(dropout)
        self.first_norm = torch.nn.BatchNorm1d(dim)
        self.attention = MHA(projection_dim, n_heads)
        self.second_dropout = torch.nn.Dropout(dropout)
        self.second_norm = torch.nn.BatchNorm1d(dim)
        self.expand = torch.nn.Linear(dim, 2*dim)
        self.contract = torch.nn.Linear(2*dim, dim)
        self.out_dropout = torch.nn.Dropout(dropout)
        self.out_norm = torch.nn.BatchNorm1d(dim)

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None):
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
        N, L, _ = Y.shape
        input = Y.reshape(N*L, -1)
        Y = self.masked_attention(Y, Y, mask=mask).reshape(N*L, -1)
        Y = self.first_norm(self.first_dropout(Y) + input).reshape(N, L, -1)
        input = Y.reshape(N*L, -1)
        Y = self.attention(Y, encoded, mask=None).reshape(N*L, -1)
        Y = self.second_norm(self.second_dropout(Y) + input)
        input = Y
        Y = self.out_dropout(self.contract(self.activation(self.expand(Y))))
        Y = self.out_norm(Y + input)
        return Y.view(N, L, -1)


class TransformerEncoder(torch.nn.Module):

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for stage in range(n_stages):
            self.stages.append(TransformerEncoderStage(projection_dim, n_heads,
                               dropout=dropout, activation=activation))

    def forward(self, X, mask=None):
        for stage in self.stages:
            X = stage(X, mask=mask)
        return X


class TransformerDecoder(torch.nn.Module):

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for stage in range(n_stages):
            self.stages.append(TransformerDecoderStage(projection_dim, n_heads,
                               dropout=dropout, activation=activation))

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None):
        for stage in self.stages:
            Y = stage(encoded, Y, mask=mask)
        return Y


class Transformer(torch.nn.Module):

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        self.encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout,
                                          activation=activation)
        self.decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout,
                                          activation=activation)

    def forward(self, X):
        return self.encode(X)

    def encode(self, X):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            tensor of embedded input tokens
            tensor of floats of shape (N, L, D) with:
            * N : number of sentences
            * L : tokens per sentence
            * D : the embedding dimension

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, L, D)
        """
        return self.encoder(X)

    def decode(self, encoded, Y, mask: Optional[torch.Tensor] = None):
        """
        performs the decoding part of the network:
        for each of the already predicted tokens, predict the next token.

        Parameters
        ----------
        encoded : torch.Tensor
            tensor of encoded inputs
            tensor of floats of shape (N, Lx, D) with:
            * N : number of sentences
            * Lx : tokens per sentence in the input language
            * D : embedding dim

        Y : torch.Tensor
            tensor of the already predicted tokens
            tensor of long of shape (N, Ly, D) with:
            * N : number of sentences
            * Ly : tokens per sentence in the output language
            * D : embedding dim

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, Ly, D)
        """
        Y = self.decoder(encoded, Y, mask=mask)
        return Y
