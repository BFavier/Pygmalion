import torch


def _align(tensor: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    """
    Truncate or repeat the last value so that 'tensor' has size 'n'
    along dimension 'dim'
    """
    L = tensor.shape[dim]
    if L < n:
        rep = (tensor.moveaxis(dim, 0)[-1]).unsqueeze(dim)
        rep = rep.expand(*(n-L if i == dim else -1 for i, _ in enumerate(rep.shape)))
        tensor = torch.cat([tensor, rep], dim=dim)
    elif L > n:
        tensor = (tensor.moveaxis(dim, 0)[:n]).moveaxis(0, dim)
    return tensor


def _mask_chronological(Lq: int, Lk: int, device: torch.device, future_offset: int=0) -> torch.Tensor:
    """
    A mask for transformers attention

    Parameters
    ----------
    Lq : int
        the sequence length of queries
    Lk : int
        the sequence length of keys
    device : torch.device
        the device to store the mask tensor on

    Returns
    -------
    torch.Tensor :
        tensor of booleans of shape (Lq, Lk)
    """
    mask = torch.ones(Lq, Lk, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1+future_offset)
    return mask


def _log_exp_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    a default kernel function for kernelized attention
    """
    return torch.log(1 + torch.exp(x))
