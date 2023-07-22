import torch
from typing import Optional, Callable
from ._utilities import _mask_chronological, _log_exp_kernel


class FourrierKernelAttention(torch.nn.Module):

    def __init__(self, kernel_function: Callable = _log_exp_kernel,
                 linear_compelxity: bool = True,
                 scaled: bool = True):
        """
        Parameters
        ----------
        kernel_function : Callable
            the kernel function applied to query and keys
        linear_complexity : bool
            whether to use linear or quadratic complexity algorithm
        scaled: bool
            if True, the scores sum up to 1
        """
        super().__init__()
        self.kernel_function = kernel_function
        self.linear_complexity = linear_compelxity
        self.scaled = scaled
    
    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor, mask_future: bool,
                padding_mask: Optional[torch.Tensor],
                pq: torch.Tensor, pk: torch.Tensor,
                mask_index_offset: int = 0):
        """
        Parameters
        ----------
        q : torch.Tensor
            query tensor of shape (N, H, Lq, D)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, D)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, D)
        mask_future : bool
            whether or not a query at index i can't attend to keys at index j > i
            in the sequence 
        padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, Lk)
        pq : torch.Tensor
            query positions, tensor of shape (N, Lq, P)
        pk : torch.Tensor
            key positions, tensor of shape (N, Lq, P)
        mask_index_offset : int
            Add the given offset to the query positions for future masking.
            This is intended for evaluation mode, where representation of
            previously generated tokens must not be generated several times.
            If different from 0, the squared complexity algorithm is used
            (because this is intended for use with a sequence of queries of length 1).

        Returns
        -------
        torch.Tensor:
            attention, a tensor of shape (N, H, Lq, D)
        """
        return self._fourrier_kernel_attention_naive(
                self.kernel_function, q, k, v, mask_future,
                padding_mask, pq, pk, self.scaled, mask_index_offset)

    @staticmethod
    def _fourrier_kernel_attention_naive(kernel: Callable, q: torch.Tensor, k: torch.Tensor,
                                         v: torch.Tensor, mask_future: bool,
                                         padding_mask: Optional[torch.Tensor],
                                         pq: torch.Tensor, pk: torch.Tensor, scaled: bool,
                                         mask_index_offset: int=0) -> torch.Tensor:
        """
        see forward doc
        Parameters
        ----------

        """
        pq, pk = kernel(q), kernel(k)
        N, H, Lq, d = pq.shape
        N, H, Lk, d = pk.shape
        sin_delta_p = torch.sin(pq.reshape(N, H, Lq, 1) - pk.reshape(N, H, 1, Lk))
        score = torch.einsum("nhqd, nhkd, nqkd -> nhqk", q, k, sin_delta_p)
        if mask_future:
            mask = _mask_chronological(Lq, Lk, score.device, mask_index_offset).reshape(1, 1, Lq, Lk)
            score = torch.masked_fill(score, mask, 0)
        if padding_mask is not None:
            score = torch.masked_fill(score, padding_mask.reshape(N, 1, 1, Lk), 0)
        if scaled:
            score = score / (torch.abs(score.sum(dim=-1).unsqueeze(-1)) + 1.0E-3)
        if padding_mask is not None:
            score = torch.masked_fill(score, padding_mask.reshape(N, 1, 1, Lk), 0.)
        attention = torch.matmul(score, v)
        return attention