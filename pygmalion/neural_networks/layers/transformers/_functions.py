import torch
from typing import Optional, Callable


def _scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor,
                                  v: torch.Tensor, mask: Optional[torch.Tensor],
                                  padding_mask: Optional[torch.Tensor],
                                  RPE: Optional[torch.nn.Embedding]
                                  ) -> torch.Tensor:
    """
    Apply scaled dot product attention to a batch of 'N' sentences pairs,
    with 'H' the number of heads, and 'D' the projection dimension.
    The query is a sequence of length 'Lq', and the key is
    a sequence of length 'Lk'.
    This is the original attention mechanism described in the 2017 paper:
        'Attention is all you need'
        https://arxiv.org/pdf/1706.03762.pdf

    Parameters
    ----------
    q : torch.Tensor
        query tensor of shape (N, H, Lq, D)
    k : torch.Tensor
        key tensor of shape (N, H, Lk, D)
    v : torch.Tensor
        value tensor of shape (N, H, Lk, D)
    mask : torch.Tensor or None
        tensor of booleans of shape (Lq, Lk)
    padding_mask : torch.Tensor or None
        tensor of booleans of shape (N, Lk)
    RPE : torch.nn.Embedding or None
        if provided, the the relative positional embedding

    Returns
    -------
    torch.Tensor:
        attention, a tensor of shape (N, H, Lq, D)
    """
    N, H, Lq, d = q.shape
    N, H, Lk, d = k.shape
    scaling = Lk**0.5 if padding_mask is None else (~padding_mask).float().sum(dim=-1).reshape(N, 1, 1, 1)**0.5
    score = torch.einsum("nhqd, nhkd -> nhqk", q, k) / scaling
    if RPE is not None:
        r = RPE.weight.shape[0] // 2
        P = torch.clip(r + torch.arange(Lk, device=score.device).reshape(1, Lk)
                       - torch.arange(Lq, device=score.device).reshape(Lq, 1), 0, 2*r)
        P = RPE(P)
        score = score + torch.einsum("qkd, nhkd -> nhqk", P, k) / scaling
    if mask is not None:
        score = score.masked_fill(mask, -float("inf"))
    if padding_mask is not None:
        score = score.masked_fill(padding_mask.reshape(N, 1, 1, Lk), -float("inf"))
    score = torch.softmax(score, dim=-1)
    attention = torch.matmul(score, v)
    return attention


def _kernelized_attention_naive(kernel: Callable, q: torch.Tensor, k: torch.Tensor,
                                v: torch.Tensor, mask: Optional[torch.Tensor],
                                padding_mask: Optional[torch.Tensor],
                                RPE: Optional[torch.nn.Embedding],
                                scaled: bool = True
                                ) -> torch.Tensor:
    """
    Parameters
    ----------
    q : torch.Tensor
        query tensor of shape (N, H, Lq, D)
    k : torch.Tensor
        key tensor of shape (N, H, Lk, D)
    v : torch.Tensor
        value tensor of shape (N, H, Lk, D)
    mask : torch.Tensor or None
        tensor of booleans of shape (Lq, Lk)
    padding_mask : torch.Tensor or None
        tensor of booleans of shape (N, Lk)
    RPE : torch.nn.Embedding or None
        if provided, the the relative positional embedding
    scaled: bool
        if True, the scores sum up to 1

    Returns
    -------
    torch.Tensor:
        attention, a tensor of shape (N, H, Lq, D)
    """
    pq, pk = kernel(q), kernel(k)
    N, H, Lq, d = pq.shape
    N, H, Lk, d = pk.shape
    score = torch.einsum("nhqd, nhkd -> nhqk", pq, pk)
    if RPE is not None:
        r = RPE.weight.shape[0] // 2
        P = torch.clip(r + torch.arange(Lk, device=score.device).reshape(1, Lk)
                       - torch.arange(Lq, device=score.device).reshape(Lq, 1), 0, 2*r)
        score = score + torch.einsum("qkd, nhqd -> nhqk", kernel(RPE(P)), pq)
    if mask is not None:
        score = score.masked_fill(mask.reshape(1, 1, Lq, Lk), 0)
    if padding_mask is not None:
        score = score.masked_fill(padding_mask.reshape(N, 1, 1, Lk), 0)
    if scaled:
        score = score / score.sum(dim=-1).unsqueeze(-1)
    attention = torch.matmul(score, v)
    return attention


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


def _kernelized_attention_linear(kernel: Callable, q: torch.Tensor, k: torch.Tensor,
                                 v: torch.Tensor, mask: bool,
                                 padding_mask: Optional[torch.Tensor],
                                 RPE: Optional[torch.nn.Embedding],
                                 scaled: bool=True) -> torch.Tensor:
    """
    Parameters
    ----------
    q : torch.Tensor
        query tensor of shape (N, H, Lq, D)
    k : torch.Tensor
        key tensor of shape (N, H, Lk, D)
    v : torch.Tensor
        value tensor of shape (N, H, Lk, D)
    mask : bool
        if True, performs masked attention
    padding_mask : torch.Tensor or None
        tensor of booleans of shape (N, Lk)
    RPE : torch.nn.Embedding or None
        if provided, the relative positional embedding

    Returns
    -------
    torch.Tensor:
        attention, a tensor of shape (N, H, Lq, D)
    """
    pq, pk = kernel(q), kernel(k)
    rpe = kernel(RPE.weight) if RPE is not None else rpe
    if padding_mask is not None:
        raise NotImplementedError()
    N, H, Lq, _ = pq.shape
    N, H, Lk, _ = pk.shape
    D = v.shape[-1]
    if mask:
        expanded = torch.einsum("nhkd, nhkD -> nhkdD", pk, v)
        summed = _align(torch.cumsum(expanded, dim=2), Lq, 2)
        attention = torch.einsum("nhqd, nhqdD -> nhqD", pq, summed)
    else:
        right = torch.einsum("nhkd, nhkD -> nhdD", pk, v)
        attention = torch.einsum("nhqd, nhdD -> nhqD", pq, right)
    if rpe is not None:
        r = rpe.shape[0] // 2
        if mask:
            rpe = torch.masked_fill(rpe, torch.arange(2*r+1).unsqueeze(-1) > r, 0.)
        W = torch.einsum("nhqd, Rd -> nhqR", pq, rpe)
        # before horizon
        p_before, n_before = min(r, Lq), min(max(0, Lq-r), Lk)
        W_before = W[..., 0]  # (N, H, Lq)
        padding_before = tuple(p_before if i == 2 else s for i, s in enumerate(v.shape))
        V_before = _align(torch.cat([torch.zeros(padding_before), v[..., :n_before, :].cumsum(dim=2)], dim=2), Lq, 2)
        attention = attention + torch.einsum("nhq, nhqd -> nhqd", W_before, V_before)
        # horizon
        W_horizon = W[..., 1:-1]  # (N, H, Lq, 2r-1)
        V_horizon = torch.cat([torch.zeros((N, H, max(0, r-1), D),
                                           device=pq.device),
                               v,
                               torch.zeros((N, H, max(0, Lq-(Lk-r)), D),
                                           device=pq.device)],
                               dim=-2)
        L = V_horizon.shape[-2]
        V_horizon = V_horizon.as_strided(size=(N, H, Lq, 2*r-1, D),
                                         stride=(H*L*D, L*D, D, D, 1))
        attention = attention + torch.einsum("nhqr, nhqrd -> nhqd", W_horizon, V_horizon)
        # after horizon
        if not mask:
            n_after = min(Lq+r, Lk)
            p_after = max(0, Lq-max(0, Lk-r))
            W_after = W[..., -1]  # (N, H, Lq)
            padding_after = torch.zeros((N, H, p_after, D), device=pq.device)
            Rcum = (v[..., r-1:n_after, :].sum(dim=-2).unsqueeze(-2)
                    - v[..., r-1:n_after-1, :].cumsum(dim=-2))
            V_after = torch.cat([Rcum, padding_after], dim=-2)
            attention = attention + torch.einsum("nhq, nhqd -> nhqd", W_after, V_after)
    if scaled:
        scale = _kernelized_attention_linear(
            kernel, q, k, torch.ones(N, H, Lk, 1), mask, padding_mask, RPE, scaled=False)
        attention = attention / scale
    return attention


def _sinusoidal_positional_encoding(X: torch.Tensor) -> torch.Tensor:
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


def _mask_chronological(Lq: int, Lk: int, device: torch.device) -> torch.Tensor:
    """
    A mask for transformers training.
    """
    mask = torch.ones(Lq, Lk, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask