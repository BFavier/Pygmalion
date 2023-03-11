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


def _mask_chronological(Lq: int, Lk: int, device: torch.device) -> torch.Tensor:
    """
    A mask for transformers attention
    """
    mask = torch.ones(Lq, Lk, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def _log_exp_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    a default kernel function for kernelized attention
    """
    return torch.log(1 + torch.exp(x))