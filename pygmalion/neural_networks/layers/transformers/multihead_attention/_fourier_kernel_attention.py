import torch
from typing import Optional, Callable
from ._utilities import _align, _mask_chronological, _log_exp_kernel


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
        self.position_weight = torch.nn.parameter.Parameter(torch.rand(self.n_heads, self.position_dimension, self.projection_dim)*2 - 1)
        self.position_bias = torch.nn.parameter.Parameter(torch.rand(self.n_heads, self.projection_dim)*2 - 1)
        self.kernel_function = kernel_function
        self.linear_complexity = linear_complexity
        self.scaled = scaled

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                history : Optional[dict] = None,
                query_mask: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None,
                query_positions: Optional[torch.Tensor] = None,
                key_positions: Optional[torch.Tensor] = None):
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
        history : dict or None
            A dict of historized key tensors or None
        query_mask : torch.Tensor or None
            Tensor of booleans of shape (N, Lq)
            or None if tokens should not be masked.
            Masked queries are set to 0 vector after transformation.
        key_mask : torch.Tensor or None
            Tensor of booleans of shape (N, Lk)
            or None if tokens should not be masked.
            Attention scores to masked keys is set to 0
        query_positions : torch.Tensor or None
            Tensor of query positions of shape (N, Lq, P)
        key_positions : torch.Tensor or None
            Tensor of query positions of shape (N, Lk, P)
        query_offset : int
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
        # project into 'n_heads' different subspaces
        q = self.kernel_function(self.query(query).reshape(N, Lq, self.n_heads, self.projection_dim))
        k = self.kernel_function(self.key(key).reshape(N, Lk, self.n_heads, self.projection_dim))
        v = self.value(key).reshape(N, Lk, self.n_heads, self.projection_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # offset
        query_offset = 0 if history is None else history.get("query_offset", 0)
        # get positions
        if query_positions is None:
            query_positions = torch.arange(query_offset, Lq+query_offset, dtype=query.dtype, device=query.device).reshape(1, Lq, 1).expand(N, -1, self.position_dimension)
        if key_positions is None:
            key_positions = torch.arange(Lk, dtype=key.dtype, device=key.device).reshape(1, Lk, 1).expand(N, -1, self.position_dimension)
        pq = (torch.einsum("nlp, hpd -> nhld", query_positions, self.position_weight)
              + torch.sigmoid(self.position_bias.reshape(1, self.n_heads, 1, self.projection_dim)) * torch.pi)
        pk = torch.einsum("nlp, hpd -> nhld", key_positions, self.position_weight)
        # append history to keys and vice versa
        if history is not None:
            K = history.get("key")
            if K is not None:
                k = torch.cat([K.to(k.device), k], dim=2)
            V = history.get("value")
            if V is not None:
                v = torch.cat([V.to(v.device), v], dim=2)
            PK = history.get("key_position")
            if PK is not None:
                pk = torch.cat([PK.to(pk.device), pk], dim=2)
            history["key"] = k
            history["value"] = v
            history["key_position"] = pk
            history["query_offset"] = query_offset + q.shape[2]
        # compute attention
        if self.linear_complexity:
            attention = self._attention_linear(
                q, k, v, pq, pk, self.mask_future, key_mask, self.scaled, query_offset)
        else:
            attention = self._attention_naive(
                q, k, v, pq, pk, self.mask_future, key_mask, self.scaled, query_offset)
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
                          pq: torch.Tensor, pk: torch.Tensor,
                          mask_future: bool, key_mask: Optional[torch.Tensor], scaled: bool,
                          query_offset: int) -> torch.Tensor:
        """
        see self._attention_naive doc's
        """
        N, H, Lq, d = q.shape
        N, H, Lk, d = k.shape
        if key_mask is not None:
            k = torch.masked_fill(k, key_mask.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, d).to(k.device), 0.)
        attention = (FourrierKernelAttention._compute_cos_sin(q, pq, k, pk, v, torch.cos, mask_future, query_offset)
                     + FourrierKernelAttention._compute_cos_sin(q, pq, k, pk, v, torch.sin, mask_future, query_offset)
                     + FourrierKernelAttention._compute(q, k, v, mask_future, query_offset))
        if scaled:
            attention = attention / (FourrierKernelAttention._compute_cos_sin(q, pq, k, pk, None, torch.cos, mask_future, query_offset)
                                     + FourrierKernelAttention._compute_cos_sin(q, pq, k, pk, None, torch.sin, mask_future, query_offset)
                                     + FourrierKernelAttention._compute(q, k, None, mask_future, query_offset)).unsqueeze(-1)
        return attention

    @staticmethod
    def _attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         pq: torch.Tensor, pk: torch.Tensor,
                         mask_future: bool, key_mask: Optional[torch.Tensor], scaled: bool,
                         query_offset: int) -> torch.Tensor:
        """
        Parameters
        ----------
        q : torch.Tensor
            query tensor of shape (N, H, Lq, d)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, d)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, d)
        pq : torch.Tensor
            tensor of query projected positions of shape (N, H, Lq, d)
        pk : torch.Tensor
            tensor of key projected positions of shape (N, H, Lk, d)
        mask_future : bool
            whether or not a query at index i can't attend to keys at index j > i
            in the sequence 
        key_mask : torch.Tensor or None
            tensor of booleans of shape (N, Lk)
        query_offset : int
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
        score = (torch.einsum("nhqd, nhkd, nhqd, nhkd -> nhqk", q, k, cos_pq, cos_pk)
                 + torch.einsum("nhqd, nhkd, nhqd, nhkd -> nhqk", q, k, sin_pq, sin_pk)
                 + torch.einsum("nhqd, nhkd -> nhqk", q, k))
        if mask_future:
            mask = _mask_chronological(Lq, Lk, score.device, query_offset).reshape(1, 1, Lq, Lk)
            score = torch.masked_fill(score, mask, 0)
        if scaled:
            score = score / score.sum(dim=-1).unsqueeze(-1)
        if key_mask is not None:
            score = torch.masked_fill(score, key_mask.reshape(N, 1, 1, Lk).to(score.device), 0.)
        attention = torch.matmul(score, v)
        return attention
    
    @staticmethod
    def _compute_cos_sin(q: torch.Tensor, pq: torch.Tensor,
                         k: torch.Tensor, pk: torch.Tensor, v: Optional[torch.Tensor],
                         cos_sin: Callable, masked: bool, query_offset: int) -> torch.Tensor:
        """
        Compute :
            \sum_k (q_{ik} * cos_sin(pq)_i * \sum_j (k_{jk} * cos_sin(pk)_j * v_{jd}))
        If masked is True, the sum over j is replaced with a cumulated sum
        If v is None, makes the denominator calculation with v_{jd} = 1.

        Parameters
        ----------
        q : torch.Tensor
            tensor of queries, post kernel function application, of shape (N, H, Lq, d)
        pq : torch.Tensor
            tensor of query positions, post projection to embedding dim, of shape (N, H, Lq, d)
        k : torch.Tensor
            tensor of queries, post kernel function application, of shape (N, H, Lk, d)
        pq : torch.Tensor
            tensor of query positions, post projection to embedding dim, of shape (N, H, Lk, d)
        v : torch.Tensor or None
            tensor of value vectors, of shape (N, H, Lk, d)
            if None, compute the denominator instead of numerator
        cos_sin : callable
            cos function or sin function
        masked : bool
            If True, compute masked future variant
        query_offset : int
            the positive offset of the query positions for the masking
            (should be 0 unless)
        """
        N, H, Lq, d = q.shape
        if v is None:
            if masked:
                right = _align(torch.cumsum(k * cos_sin(pk), dim=2), n=Lq+query_offset, dim=2)[..., query_offset:, :]
                return torch.einsum("nhqd, nhqd, nhqd -> nhq", q, cos_sin(pq), right)
            else:
                return torch.einsum("nhqd, nhqd, nhd -> nhq", q, cos_sin(pq), torch.sum(k * cos_sin(pk), 2))
        else:
            if masked:
                right = torch.einsum("nhkd, nhkd, nhkv -> nhkdv", k, cos_sin(pk), v)
                right = _align(torch.cumsum(right, dim=2), n=Lq+query_offset, dim=2)[..., query_offset:, :, :]
                return torch.einsum("nhkd, nhkd, nhkdv -> nhkv", q, cos_sin(pq), right)
            else:
                right = torch.einsum("nhkd, nhkd, nhkv -> nhdv", k, cos_sin(pk), v)
                return torch.einsum("nhqd, nhqd, nhdv -> nhqv", q, cos_sin(pq), right)

    @staticmethod
    def _compute(q: torch.Tensor, k: torch.Tensor, v: Optional[torch.Tensor],
                 masked: bool, query_offset: int) -> torch.Tensor:
        """
        Compute :
            \sum_k (q_{ik} * \sum_j (k_{jk} * v_{jd}))
        If masked is True, the sum over j is replaced with a cumulated sum
        If v is None, makes the denominator calculation with v_{jd} = 1.

        Parameters
        ----------
            See 'self._compute_cos_sin' for parameters description
        """
        N, H, Lq, d = q.shape
        if v is None:
            if masked:
                right = _align(torch.cumsum(k, dim=2), n=Lq+query_offset, dim=2)[..., query_offset:, :]
                return torch.einsum("nhqd, nhqd -> nhq", q, right)
            else:
                return torch.einsum("nhqd, nhd -> nhq", q, torch.sum(k, 2))
        else:
            if masked:
                right = torch.einsum("nhkd, nhkv -> nhkdv", k, v)
                right = _align(torch.cumsum(right, dim=2), n=Lq+query_offset, dim=2)[..., query_offset:, :, :]
                return torch.einsum("nhkd, nhkdv -> nhkv", q, right)
            else:
                right = torch.einsum("nhkd, nhkv -> nhdv", k, v)
                return torch.einsum("nhqd, nhdv -> nhqv", q, right)
