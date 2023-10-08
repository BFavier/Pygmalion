import torch
from typing import Optional, Callable
from ._utilities import _align, _mask_chronological, _log_exp_kernel

class KernelizedAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 mask_future: bool, RPE_radius: Optional[int] = None,
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
        scaled : bool
            if True, the attention scores of any query to all keys sums up to 1
        """
        super().__init__()
        self.n_heads = n_heads
        self.projection_dim = projection_dim
        dim = projection_dim * n_heads
        self.mask_future = mask_future
        self.relative_positional_encoding = torch.nn.Embedding(2*RPE_radius+1, dim) if RPE_radius else None
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)
        self.kernel_function = kernel_function
        self.linear_complexity = linear_complexity
        self.scaled = scaled

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                history : Optional[dict] = None,
                query_mask: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None):
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
        history : 
        query_mask : torch.Tensor or None
            Tensor of booleans of shape (N, Lq)
            or None if padding tokens should not be masked.
            Masked queries are set to null vector after transformation.
        key_mask : torch.Tensor or None
            Tensor of booleans of shape (N, Lk)
            or None if padding tokens should not be masked.
            Attention scores to masked keys is set to 0

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
        # append history to keys and vice versa
        query_offset = 0
        if history is not None:
            K = history.get("key")
            if K is not None:
                k = torch.cat([K.to(k.device), k], dim=2)
            V = history.get("value")
            if V is not None:
                v = torch.cat([V.to(v.device), v], dim=2)
            query_offset = history.get("query_offset", 0)
            history["key"] = k
            history["value"] = v
            history["query_offset"] = query_offset + q.shape[2]
        # compute attention
        if self.linear_complexity:
            attention = self._attention_linear(
                self.kernel_function, q, k, v, self.mask_future, key_mask, self.relative_positional_encoding, self.scaled, query_offset)
        else:
            attention = self._attention_naive(
                self.kernel_function, q, k, v, self.mask_future, key_mask, self.relative_positional_encoding, self.scaled, query_offset)
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
    def _attention_linear(kernel: Callable, Q: torch.Tensor, K: torch.Tensor,
                          v: torch.Tensor, mask_future: bool,
                          key_mask: Optional[torch.Tensor],
                          RPE: Optional[torch.nn.Embedding],
                          scaled: bool, query_offset: int = 0) -> torch.Tensor:
        """
        see self._attention_naive doc
        """
        if query_offset != 0:
            raise ValueError("Linear complexity not implemented for 'query_offset' != 0")
        q, k = kernel(Q), kernel(K)
        N, H, Lq, _ = q.shape
        N, H, Lk, _ = k.shape
        D = v.shape[-1]
        if key_mask is not None:
            v = torch.masked_fill(v, key_mask.reshape(N, 1, Lk, 1), 0.)
        if mask_future:
            expanded = torch.einsum("nhkd, nhkD -> nhkdD", k, v)
            summed = _align(torch.cumsum(expanded, dim=2), Lq, 2)
            attention = torch.einsum("nhqd, nhqdD -> nhqD", q, summed)
        else:
            right = torch.einsum("nhkd, nhkD -> nhdD", k, v)
            attention = torch.einsum("nhqd, nhdD -> nhqD", q, right)
        if RPE is not None:
            rpe = kernel(RPE.weight)
            r = rpe.shape[0] // 2
            if mask_future:
                rpe = torch.masked_fill(rpe, torch.arange(2*r+1).unsqueeze(-1) > r, 0.)
            W = torch.einsum("nhqd, Rd -> nhqR", q, rpe)
            # before horizon
            p_before, n_before = min(r, Lq), min(max(0, Lq-r), Lk)
            W_before = W[..., 0]  # (N, H, Lq)
            padding_before = tuple(p_before if i == 2 else s for i, s in enumerate(v.shape))
            V_before = _align(torch.cat([torch.zeros(padding_before), v[..., :n_before, :].cumsum(dim=2)], dim=2), Lq, 2)
            attention = attention + torch.einsum("nhq, nhqd -> nhqd", W_before, V_before)
            # horizon
            W_horizon = W[..., 1:-1]  # (N, H, Lq, 2r-1)
            V_horizon = torch.cat([torch.zeros((N, H, max(0, r-1), D),
                                            device=q.device),
                                v,
                                torch.zeros((N, H, max(0, Lq-(Lk-r)), D),
                                            device=q.device)],
                                dim=-2)
            L = V_horizon.shape[-2]
            V_horizon = V_horizon.as_strided(size=(N, H, Lq, 2*r-1, D),
                                            stride=(H*L*D, L*D, D, D, 1))
            attention = attention + torch.einsum("nhqr, nhqrd -> nhqd", W_horizon, V_horizon)
            # after horizon
            if not mask_future:
                n_after = min(Lq+r, Lk)
                p_after = max(0, Lq-max(0, Lk-r))
                W_after = W[..., -1]  # (N, H, Lq)
                padding_after = torch.zeros((N, H, p_after, D), device=q.device)
                Rcum = (v[..., r-1:n_after, :].sum(dim=-2).unsqueeze(-2)
                        - v[..., r-1:n_after-1, :].cumsum(dim=-2))
                V_after = torch.cat([Rcum, padding_after], dim=-2)
                attention = attention + torch.einsum("nhq, nhqd -> nhqd", W_after, V_after)
        if scaled:
            v_scaling = torch.ones(N, H, Lk, 1)
            if key_mask is not None:
                v_scaling = torch.masked_fill(v_scaling, key_mask.reshape(N, 1, Lk, 1), 0.)
            scale = KernelizedAttention._attention_linear(
                kernel, Q, K, v_scaling, mask_future, key_mask, RPE, scaled=False)
            attention = attention / scale
        return attention

    @staticmethod
    def _attention_naive(kernel: Callable, q: torch.Tensor, k: torch.Tensor,
                         v: torch.Tensor, mask_future: bool,
                         key_mask: Optional[torch.Tensor],
                         RPE: Optional[torch.nn.Embedding],
                         scaled: bool, query_offset: int = 0) -> torch.Tensor:
        """
        Parameters
        ----------
        kernel : callable
            kernel function
        q : torch.Tensor
            query tensor of shape (N, H, Lq, d)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, d)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, d)
        mask_future : bool
            whether or not a query at index i can't attend to keys at index j > i
            in the sequence 
        key_mask : torch.Tensor or None
            tensor of booleans of shape (N, Lk)
        RPE : torch.nn.Embedding or None
            if provided, the relative positional embedding
            tensor of shape (2*R+1, D) or None
        scaled : bool
            if True, the attention scores of any query to all keys sums up to 1
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
        q, k = kernel(q), kernel(k)
        N, H, Lq, d = q.shape
        N, H, Lk, d = k.shape
        score = torch.einsum("nhqd, nhkd -> nhqk", q, k)
        if RPE is not None:
            r = RPE.weight.shape[0] // 2
            P = torch.clip(r + torch.arange(Lk, device=score.device).reshape(1, Lk)
                           - torch.arange(Lq, device=score.device).reshape(Lq, 1)
                           - query_offset,
                           0, 2*r)
            score = score + torch.einsum("qkd, nhqd -> nhqk", kernel(RPE(P)), q)
        if mask_future:
            mask = _mask_chronological(Lq, Lk, score.device, query_offset).reshape(1, 1, Lq, Lk)
            score = torch.masked_fill(score, mask, 0)
        if key_mask is not None:
            score = torch.masked_fill(score, key_mask.reshape(N, 1, 1, Lk), 0)
        if scaled:
            score = score / score.sum(dim=-1).unsqueeze(-1)
        if key_mask is not None:
            score = torch.masked_fill(score, key_mask.reshape(N, 1, 1, Lk), 0.)
        attention = torch.matmul(score, v)
        return attention