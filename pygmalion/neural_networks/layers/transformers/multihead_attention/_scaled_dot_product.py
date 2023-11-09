import torch
from typing import Optional
from ._utilities import _mask_chronological


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 mask_future: bool, RPE_radius: Optional[int]=None):
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
        query_mask : torch.Tensor or None
            Tensor of booleans of shape (N, Lq)
            or None if padding tokens should not be masked.
            Masked queries are set to null vector after transformation.
        key_mask : torch.Tensor or None
            Tensor of booleans of shape (N, Lk) or None
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
        q = self.query(query).reshape(N, Lq, self.n_heads, self.projection_dim)
        k = self.key(key).reshape(N, Lk, self.n_heads, self.projection_dim)
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
        attention = self._attention(
            q, k, v, self.mask_future, key_mask,
            self.relative_positional_encoding,
            query_offset=query_offset)
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
    def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   mask_future: bool, padding_mask: Optional[torch.Tensor],
                   RPE: Optional[torch.nn.Embedding], query_offset: int = 0
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
            query tensor of shape (N, H, Lq, d)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, d)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, d)
        mask_future : bool
            whether or not a query at index i can't attend to keys at index j > i
            in the sequence 
        padding_mask : torch.Tensor or None
            Tensor of booleans of shape (N, Lk).
            Masked tensors (mask set to True) have their attrention set to 0.
        RPE : torch.nn.Embedding or None
            if provided, the relative positional embedding
            tensor of shape (2*R+1, D) or None
        query_offset : int
            Add the given offset to the query positions for future masking.
            This is intended for evaluation mode, where representation of
            previously generated tokens must not be generated several times.

        Returns
        -------
        torch.Tensor:
            attention, a tensor of shape (N, H, Lq, d)
        """
        N, H, Lq, d = q.shape
        N, H, Lk, d = k.shape
        score = torch.einsum("nhqd, nhkd -> nhqk", q, k) / d**0.5
        if RPE is not None:
            r = RPE.weight.shape[0] // 2
            P = torch.clip(r + torch.arange(Lk, device=score.device).reshape(1, Lk)
                           - torch.arange(Lq, device=score.device).reshape(Lq, 1)
                           - query_offset, 0, 2*r)
            P = RPE(P).reshape(Lq, Lk, H, d)
            score = score + torch.einsum("qkhd, nhkd -> nhqk", P, k) / d**0.5
        if mask_future:
            score = score.masked_fill(_mask_chronological(Lq, Lk, score.device, query_offset).reshape(1, 1, Lq, Lk), -float("inf"))
        if padding_mask is not None:
            score = score.masked_fill(padding_mask.to(score.device).reshape(N, 1, 1, Lk), -float("inf"))
        score = torch.softmax(score, dim=-1)
        attention = torch.matmul(score, v)
        return attention