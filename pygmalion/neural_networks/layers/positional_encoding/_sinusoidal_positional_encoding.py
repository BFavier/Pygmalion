import torch

class SinusoidalPositionalEncoding(torch.nn.Module):
    """
    Parameterless positional encoding for sequences
    Performs positional encoding on the input, in the
    "Attention is all you need" paper fashion.
    """

    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
    
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
        N, _ = X.shape
        D = self.embedding_dimension
        pe = torch.zeros(N, D, dtype=torch.float, device=X.device)
        position = torch.arange(0, D, dtype=torch.float).unsqueeze(0) + offset
        angle = position / 10000**(2*torch.div(position, 2, rounding_mode='floor')/D)
        pe[:, 0::2] = torch.cos(angle[:, 0::2])
        pe[:, 1::2] = torch.sin(angle[:, 1::2])
        X = (X + pe).reshape(shape)
        return X