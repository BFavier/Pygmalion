import torch


class LearnedPositionalEncoding(torch.nn.Module):
    """
    Learned positional encoding for sequences
    """

    def __init__(self, embedding_dimension: int, n_positions: int):
        """
        Parameters
        ----------
        embedding_dimension : int
            Embedding vector dimension
        n_positions : int
            Maximum length of the sequence to encode position in.
            There won't be a learned embedding vector for tokens beyond 'n_positions'.
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
        n_positions = self.embedding.weight.shape[0]
        if (L+offset >= n_positions):
            raise ValueError(f"Tried applying {type(self).__name__} with {n_positions} learned positions to longer sequence of length {L}. (Tensor of shape {tuple(X.shape)})")
        P = torch.arange(L, device=X.device)
        shape = tuple(1 for _ in range(len(X.shape) - 2)) + (L, D)
        return X + self.embedding(P+offset).reshape(shape)