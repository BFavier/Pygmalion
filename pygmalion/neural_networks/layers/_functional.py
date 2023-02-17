import torch


def sinusoidal_positional_encoding(X: torch.Tensor) -> torch.Tensor:
    """
    Performs positional encoding on the input, in the
    "Attention is all you need" paper fashion.

    Parameters
    ----------
    X : torch.Tensor
        tensor of shape (..., D), with D the embedding dimension

    Returns
    -------
    torch.Tensor:
        tensor of shape (..., D)
    """
    shape = X.shape
    X = X.reshape(-1, shape[-1])
    N, D = X.shape
    pe = torch.zeros(N, D, dtype=torch.float, device=X.device)
    position = torch.arange(0, D, dtype=torch.float).unsqueeze(0)
    angle = position / 10000**(2*(position//2)/D)
    pe[:, 0::2] = torch.cos(angle[:, 0::2])
    pe[:, 1::2] = torch.sin(angle[:, 1::2])
    X = (X + pe).reshape(shape)
    return X


def mask_chronological(Lq: int, Lk: int, device: torch.device) -> torch.Tensor:
    """
    A mask for transformers training.
    """
    mask = torch.ones(Lq, Lk, dtype=bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask
