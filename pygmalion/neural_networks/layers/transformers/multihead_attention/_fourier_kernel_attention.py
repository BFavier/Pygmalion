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
                q, k, v, self.position_coeffs, pq, pk, self.mask_future, key_mask, self.scaled)
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
                          position_coeff: torch.Tensor, pq: torch.Tensor, pk: torch.Tensor,
                          mask_future: bool, key_mask: Optional[torch.Tensor], scaled: bool
                          ) -> torch.Tensor:
        """
        see self._attention_naive doc's
        """
        N, H, Lq, d = q.shape
        N, H, Lk, d = k.shape
        if key_mask is not None:
            k = torch.masked_fill(k, key_mask.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, d).to(k.device), 0.)
        attention = (FourrierKernelAttention._compute_part(position_coeff, q, pq, k, pk, v, torch.cos, mask_future)
                     + FourrierKernelAttention._compute_part(position_coeff, q, pq, k, pk, v, torch.sin, mask_future))
        if scaled:
            unit = torch.ones(N, H, Lk, 1, device=q.device)
            attention /= (FourrierKernelAttention._compute_part(position_coeff, q, pq, k, pk, unit, torch.cos, mask_future)
                          + FourrierKernelAttention._compute_part(position_coeff, q, pq, k, pk, unit, torch.sin, mask_future))
        return attention

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
            score = score / score.sum(dim=-1).unsqueeze(-1)
        if key_mask is not None:
            score = torch.masked_fill(score, key_mask.reshape(N, 1, 1, Lk).to(score.device), 0.)
        attention = torch.matmul(score, v)
        return attention
    
    @staticmethod
    def _compute_part(c: torch.Tensor, q: torch.Tensor, pq: torch.Tensor,
                      k: torch.Tensor, pk: torch.Tensor, v: torch.Tensor,
                      cos_sin: Callable, masked: bool) -> torch.Tensor:
        """
        Compute
        \sum_k (c_k * q_{ik} * cos_sin(pq)_i * \sum_j (k_{jk} * cos_sin(pk)_j * v_{jd}))
        If masked is True, the sum over j is replaced with a cumulated sum
        
        Parameters
        ----------
        c : torch.Tensor
            position coefficient of shape (H, d)
        q : torch.Tensor
            tensor of queries, post kernel function application, of shape (N, H, Lq, d)
        pq : torch.Tensor
            tensor of query positions, post projection to embedding dim, of shape (N, H, Lq, d)
        k : torch.Tensor
            tensor of queries, post kernel function application, of shape (N, H, Lk, d)
        pq : torch.Tensor
            tensor of query positions, post projection to embedding dim, of shape (N, H, Lk, d)
        v : torch.Tensor
            tensor of value vectors, of shape (N, H, Lk, d)
        cos_sin : callable
            cos function or sin function
        masked : bool
            If True, compute masked future variant
        """
        N, H, Lq, d = q.shape
        N, H, Lk, d = k.shape
        if masked:
            right = torch.einsum("nhjk, nhjk, nhjd -> nhjkd", k, cos_sin(pk), v)
            right = right[:, :, :Lq, ...]
            if Lq > Lk:
                right = torch.cat([right, right[:, :, -1:, ...].expand(-1, -1, Lq-Lk, -1, -1)], dim=2)
            right = torch.cumsum(right, dim=2)
            left = torch.einsum("hk, nhik, nhik, nhikd -> nhid", c, q, cos_sin(pq), right)
        else:
            right = torch.einsum("nhjk, nhjk, nhjd -> nhkd", k, cos_sin(pk), v)
            left = torch.einsum("hk, nhik, nhik, nhkd -> nhid", c, q, cos_sin(pq), right)
        return left