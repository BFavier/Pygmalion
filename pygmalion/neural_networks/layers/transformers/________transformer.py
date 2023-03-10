import torch
from typing import Optional
from ._stages import TransformerEncoder, TransformerDecoder

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
