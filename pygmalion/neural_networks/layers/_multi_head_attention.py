import torch
import math
from ._weighting import Linear
from ._batch_norm import BatchNorm0d
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None):
        super().__init__()
        self.n_heads = n_heads
        self.projection_dim = projection_dim
        self.dropout = dropout
        dim = projection_dim*n_heads
        self.query = Linear(dim, dim, bias=False)
        self.key = Linear(dim, dim, bias=False)
        self.value = Linear(dim, dim, bias=False)
        self.norm = BatchNorm0d(dim)

    @classmethod
    def from_dump(cls, dump: dict) -> 'MultiheadAttention':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.n_heads = dump["n heads"]
        obj.projection_dim = dump["projection dim"]
        obj.query = Linear.from_dump(dump["query"])
        obj.key = Linear.from_dump(dump["key"])
        obj.value = Linear.from_dump(dump["value"])
        obj.norm = BatchNorm0d.from_dump(dump["norm"])
        obj.dropout = dump["dropout"]
        return obj

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the multihead attention module.
        Apply masked attention, followed by dropout, and batch normalization

        Parameters
        ----------
        query : torch.Tensor
            tensor of shape (N, Lq, D) with
            * N the number of sentences to treat
            * Lq the sequence length of the query
            * D the embedding dimension
        key : torch.Tensor
            tensor of shape (N, Lk, D) with
            * N the number of sentences to treat
            * Lk the sequence length of the key
            * D the embedding dimension
        mask : torch.Tensor or None
            the mask, tensor of booleans of shape (Lq, Lk), where attention
            is set to -infinity

        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Lq, D)
        """
        attention = self._multihead_attention(query, key, mask)
        if self.dropout is not None:
            attention = F.dropout(attention, p=self.dropout,
                                  training=self.training)
        N, Lq, D = attention.shape
        attention = attention + query
        attention = self.norm(attention.view(-1, self.out_features))
        return attention.view(N, Lq, self.out_features)

    def _multihead_attention(self, query: torch.Tensor, key: torch.Tensor,
                             mask: Optional[torch.Tensor]):
        """
        Apply multihead attention.
        Same inputs/outputs types/shapes as the forward pass
        """
        N, Lq, _ = query.shape
        N, Lk, _ = key.shape
        q = self.query(query).view(N, Lq, self.n_heads, self.projection_dim)
        k = self.key(key).view(N, Lk, self.n_heads, self.projection_dim)
        v = self.value(key).view(N, Lk, self.n_heads, self.projection_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attention, _ = self._scaled_dot_product_attention(q, k, v, mask)
        attention = attention.transpose(2, 1).reshape(N, Lq, -1)
        return attention

    def _scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor,
                                      v: torch.Tensor,
                                      mask: Optional[torch.Tensor]
                                      ) -> torch.Tensor:
        """
        Apply scaled dot product attention to a batch of 'N' observations, with
        'H' the number of heads, and 'D' the projection dimension.
        The query is a sequence of length 'Lq', and the key is
        a sequence of length 'Lk'.

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

        Returns
        -------
        tuple of torch.Tensors:
            a tuple of (attention, score)
        """
        Lk = k.shape[-1]
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Lk)
        if mask is not None:
            score = score.masked_fill(mask, -1.0E6)
        score = torch.softmax(score, dim=-1)
        attention = torch.matmul(score, v)
        return attention, score

    def _multihead_attention_stock(self, query: torch.Tensor,
                                   key: torch.Tensor,
                                   mask: Optional[torch.Tensor]):
        """
        Apply multihead attention.
        Same inputs/outputs types/shapes as the forward pass
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = key
        embed_dim = self.projection_dim*self.n_heads
        in_proj_weight = None
        in_proj_bias = None
        out_proj_weight = torch.eye(embed_dim, dtype=torch.float,
                                    device=query.device)
        out_proj_bias = torch.zeros(embed_dim, dtype=torch.float,
                                    device=query.device)
        dropout = 0.
        return F.multi_head_attention_forward(
                query, key, value, embed_dim, self.n_heads,
                in_proj_weight, in_proj_bias,
                None, None, False,
                dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=None, need_weights=False,
                attn_mask=mask, use_separate_proj_weight=True,
                q_proj_weight=self.query.weight,
                k_proj_weight=self.key.weight,
                v_proj_weight=self.value.weight)[0].transpose(1, 0)

    @property
    def in_features(self):
        return self.query.in_features

    @property
    def out_features(self):
        return self.value.out_features

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "n heads": self.n_heads,
                "projection dim": self.projection_dim,
                "query": self.query.dump,
                "key": self.key.dump,
                "value": self.value.dump,
                "norm": self.norm.dump,
                "dropout": self.dropout}
