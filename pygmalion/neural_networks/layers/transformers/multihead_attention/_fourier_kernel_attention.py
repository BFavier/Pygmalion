import torch
from typing import Optional, Callable
from ._utilities import _mask_chronological, _log_exp_kernel


class FourrierKernelAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 mask_future: bool, position_dimension: int = 1,
                 kernel_function: Callable = _log_exp_kernel,
                 linear_complexity: bool = True,
                 scaled: bool = True):
        """
        Parameters
        ----------
        projection_dim : int
            the dimension of the projection space for the feature vectors
        n_heads : int
            the number of different projection at each stage of the transformer
        mask_future: bool
            whether or not a query at index i can't attend to keys
            at index j > i in the sequence
        RPE_radius : int or None
            The radius of the relative positional encoding
            or None if no relative positional encoding should be applied
        kernel_function : Callable
            the kernel function applied to query and keys
        linear_complexity : bool
            whether to use linear or quadratic complexity algorithm
        scaled: bool
            if True, the scores sum up to 1
        """
        super().__init__()
        self.n_heads = n_heads
        self.projection_dim = projection_dim
        self.position_dimension = position_dimension
        dim = projection_dim * n_heads
        self.mask_future = mask_future
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)
        self.position_coeffs = torch.nn.parameter.Parameter(torch.ones(self.n_heads, self.projection_dim))
        self.position_weight = torch.nn.parameter.Parameter(torch.zeros(self.n_heads, self.projection_dim, self.position_dimension))
        self.position_bias = torch.nn.parameter.Parameter(torch.zeros(self.n_heads, self.projection_dim))
        self.kernel_function = kernel_function
        self.linear_complexity = linear_complexity
        self.scaled = scaled

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None,
                query_positions: Optional[torch.Tensor] = None,
                key_positions: Optional[torch.Tensor] = None,
                future_offset: int=0):
        """
        Apply scaled dot product attention to a batch of 'N' sentences pairs,
        with 'H' the number of heads, and 'D' the projection dimension.
        The query is a sequence of length 'Lq', and the key is
        a sequence of length 'Lk'.
        This is the original attention mechanism described in the 2017 paper:
            'Attention is all you need'
            https://arxiv.org/abs/1706.03762
        In addition, relative positional encoding is implemented as well:
            'Self-Attention with Relative Position Representations'
            https://arxiv.org/abs/1803.02155

        Parameters
        ----------
        query : torch.Tensor
            tensor of shape (N, Lq, D)
        key : torch.Tensor
            tensor of shape (N, Lk, D)
        query_mask : torch.Tensor or None
            Tensor of booleans of shape (N, Lq)
            or None if tokens should not be masked.
            Masked queries are set to null vector after transformation.
        key_mask : torch.Tensor or None
            Tensor of booleans of shape (N, Lk)
            or None if tokens should not be masked.
            Attention scores to masked keys is set to 0
        query_positions : torch.Tensor or None
            Tensor of query positions of shape (N, Lq, P)
        key_positions : torch.Tensor or None
            Tensor of query positions of shape (N, Lk, P)
        future_offset : int
            Add the given offset to the query positions for future masking.
            This is intended for evaluation mode, where representation of
            previously generated tokens must not be generated several times.

        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Lq, D)
        """
        query, key = query.to(self.device), key.to(self.device)
        N, Lq, _ = query.shape
        N, Lk, _ = key.shape
        # set default positions
        if query_positions is None:
            query_positions = torch.arange(Lq, dtype=query.dtype, device=query.device).reshape(1, Lq, 1).expand(N, -1, self.position_dimension)
        if key_positions is None:
            key_positions = torch.arange(Lk, dtype=key.dtype, device=key.device).reshape(1, Lk, 1).expand(N, -1, self.position_dimension)
        # project into 'n_heads' different subspaces
        q = self.kernel_function(self.query(query).reshape(N, Lq, self.n_heads, self.projection_dim))
        k = self.kernel_function(self.key(key).reshape(N, Lk, self.n_heads, self.projection_dim))
        v = self.value(key).reshape(N, Lk, self.n_heads, self.projection_dim)
        pq = (torch.einsum("nlp, hkp -> nhlk", query_positions, self.position_weight)
              + self.position_bias.reshape(1, self.n_heads, 1, self.projection_dim))
        pk = torch.einsum("nlp, hkp -> nhlk", query_positions, self.position_weight)
        # compute attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.linear_complexity and (future_offset == 0):
            attention = self._attention_linear(
                q, k, v, self.position_coeffs, pq, pk, self.mask_future, key_mask, self.scaled, future_offset)
        else:
            attention = self._attention_naive(
                q, k, v, self.position_coeffs, pq, pk, self.mask_future, key_mask, self.scaled, future_offset)
        attention = attention.transpose(2, 1).reshape(N, Lq, -1)
        # mask queries if needed
        if query_mask is not None:
            query_mask = query_mask.to(attention.device).unsqueeze(-1)
            attention = torch.masked_fill(attention, query_mask, 0.)
        return attention

    @property
    def device(self) -> torch.device:
        return self.query.weight.device

    @staticmethod
    def _attention_linear(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          pq: torch.Tensor, pk: torch.Tensor, mask_future: bool,
                          key_mask: Optional[torch.Tensor], scaled: bool,
                          future_offset: int) -> torch.Tensor:
        """
        see self._attention_naive doc's
        """
        ...
        raise NotImplementedError()

    @staticmethod
    def _attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         position_coeff: torch.Tensor, pq: torch.Tensor, pk: torch.Tensor,
                         mask_future: bool, key_mask: Optional[torch.Tensor], scaled: bool,
                         future_offset: int) -> torch.Tensor:
        """
        Parameters
        ----------
        q : torch.Tensor
            query tensor of shape (N, H, Lq, d)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, d)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, d)
        position_coeff : torch.Tensor
            coeff of shape (H, d)
        pq : torch.Tensor
            tensor of query projected positions of shape (N, H, Lq, d)
        pk : torch.Tensor
            tensor of key projected positions of shape (N, H, Lk, d)
        mask_future : bool
            whether or not a query at index i can't attend to keys at index j > i
            in the sequence 
        key_mask : torch.Tensor or None
            tensor of booleans of shape (N, Lk)
        future_offset : int
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
        N, H, Lq, d = q.shape
        N, H, Lk, d = k.shape
        cos_pq, cos_pk = torch.cos(pq), torch.cos(pk)
        sin_pq, sin_pk = torch.sin(pq), torch.sin(pk)
        score = (torch.einsum("nhqd, nhkd, hd, nhqd, nhkd -> nhqk", q, k, position_coeff, cos_pq, cos_pk)
                 + torch.einsum("nhqd, nhkd, hd, nhqd, nhkd -> nhqk", q, k, position_coeff, sin_pq, sin_pk))
        if mask_future:
            mask = _mask_chronological(Lq, Lk, score.device, future_offset).reshape(1, 1, Lq, Lk)
            score = torch.masked_fill(score, mask, 0)
        if scaled:
            score = score / score.abs().sum(dim=-1).unsqueeze(-1)
        if key_mask is not None:
            score = torch.masked_fill(score, key_mask.reshape(N, 1, 1, Lk).to(score.device), 0.)
        attention = torch.matmul(score, v)
        return attention