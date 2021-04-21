import torch
import math
from ._weighting import Linear
from ._batch_norm import BatchNorm0d


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.projection_dim = projection_dim
        dim = projection_dim*n_heads
        self.query = Linear(dim, dim, bias=False)
        self.key = Linear(dim, dim, bias=False)
        self.value = Linear(dim, dim, bias=False)
        self.norm = BatchNorm0d(dim)

    @classmethod
    def from_dump(cls, dump: dict) -> 'MultiHeadAttention':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.n_heads = dump["n heads"]
        obj.projection_dim = dump["projection dim"]
        obj.query = Linear.from_dump(dump["query"])
        obj.key = Linear.from_dump(dump["key"])
        obj.value = Linear.from_dump(dump["value"])
        obj.norm = BatchNorm0d.from_dump(dump["norm"])
        return obj

    def forward(self, query, key, masked: bool = True):
        """
        Parameters
        ----------
        query : torch.Tensor
            tensor of input words of shape (N, Lq, D) with
            * N the number of sentences to treat
            * Lq the number of words per input sentence
            * D the input embeding dimension of each word


        key : torch.Tensor
            tensor of predicted output words of shape (N, Lk, D) with
            * N the number of sentences to treat
            * Lk the number of words per output sentence
            * D the output embeding dimension of each word

        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Ly, P*H), with :
            * P the projection dimension
            * H the number of heads
        """
        N, Lq, _ = query.shape
        N, Lk, _ = key.shape
        q = self.query(query).view(N, Lq, 1, self.n_heads, self.projection_dim)
        k = self.key(key).view(N, 1, Lk, self.n_heads, self.projection_dim)
        v = self.value(key).view(N, 1, Lk, self.n_heads, self.projection_dim)
        score = torch.sum(q*k, dim=-1) / math.sqrt(self.out_channels)
        if masked:
            mask = torch.ones(1, Lq, Lk, dtype=bool, device=score.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(-1)
            score = score.masked_fill(mask, -1.0E6)
        attention = torch.softmax(score, dim=2).unsqueeze(-1)
        res = torch.sum(v*attention, dim=2)
        res = res.view(N, Lq, -1) + query
        res = self.norm(res.view(-1, self.out_channels))
        return res.view(N, Lq, self.out_channels)

    @property
    def in_channels(self):
        return self.query.in_channels

    @property
    def out_channels(self):
        return self.value.out_channels

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "n heads": self.n_heads,
                "projection dim": self.projection_dim,
                "query": self.query.dump,
                "key": self.key.dump,
                "value": self.value.dump,
                "norm": self.norm.dump}
