import torch


class LearnedPositionalEncoding(torch.nn.Module):
    """
    Learned positional encoding for sequences
    """

    def __init__(self, n_positions: int, embedding_dimension: int):
        """
        Parameters
        ----------
        n_positions : int
            maximum length of the sequence to encode position in
            There won't be a learned embedding vector for tokens beyond 'n_positions'
        embedding_dimension : int
            Embedding vector dimension
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(n_positions, embedding_dimension)

    def forward(self, X: torch.Tensor, offset: int=0) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (..., L, D)
        offset : int
            a position offset
        
        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (..., L, D)
        """
        L, D = X.shape[-2:]
        P = torch.arange(L, device=X.device)
        shape = tuple(1 for _ in range(len(X.shape) - 2)) + (L, D)
        return X + self.embedding(P+offset).reshape(shape)


class SinusoidalPositionalEncoding(torch.nn.Module):
    """
    Parameterless positional encoding for sequences
    Performs positional encoding on the input, in the
    "Attention is all you need" paper fashion.
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, X: torch.Tensor, offset: int=0) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (..., D), with D the embedding dimension
        offset : int
            a position offset

        Returns
        -------
        torch.Tensor:
            tensor of shape (..., D)
        """
        shape = X.shape
        X = X.reshape(-1, shape[-1])
        N, D = X.shape
        pe = torch.zeros(N, D, dtype=torch.float, device=X.device)
        position = torch.arange(0, D, dtype=torch.float).unsqueeze(0) + offset
        angle = position / 10000**(2*torch.div(position, 2, rounding_mode='floor')/D)
        pe[:, 0::2] = torch.cos(angle[:, 0::2])
        pe[:, 1::2] = torch.sin(angle[:, 1::2])
        X = (X + pe).reshape(shape)
        return X