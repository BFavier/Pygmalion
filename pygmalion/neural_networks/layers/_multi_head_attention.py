import torch
import math
from ._weighting import Linear


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, n_heads: int):
        super().__init__()
        self._n_heads = n_heads
        self.query = Linear(in_features, out_features*n_heads, bias=False)
        self.key = Linear(in_features, out_features*n_heads, bias=False)
        self.value = Linear(in_features, out_features*n_heads, bias=False)

    @classmethod
    def from_dump(cls, dump: dict) -> 'MultiHeadAttention':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.query = Linear.from_dump(dump["query"])
        obj.key = Linear.from_dump(dump["key"])
        obj.value = Linear.from_dump(dump["value"])
        return obj

    def forward(self, X, Y, masked: bool = True):
        """
        Parameters
        ----------
        X : torch.Tensor
            input of shape (N, L, I) with
            * N the number of sentences to treat
            * L the number of words per sentence
            * I the input embeding dimension of each word

        Returns
        -------
        torch.Tensor :
            tensor of shape (N, L, H, O), with :
            * H the number of heads
            * O the output embeding dimension
        """
        N, L, _ = X.shape
        q = self.query(X)
        k = self.key(Y)
        v = self.value(Y)
        q = q.unsqueeze(2).expand(-1, -1, self.out_channels, -1)
        k = k.unsqueeze(3).expand(-1, -1, -1, self.out_channels)
        score = torch.sum(q*k, dim=-1) / math.sqrt(self.out_channels)
        score = score.view(N, L, self.n_heads, self.out_channels)
        if masked:
            mask = torch.triu(torch.ones(N, L, dtype=bool), diagonal=1)
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            mask = mask.expand(-1, -1, self.n_heads, self.out_channels)
            score = score.masked_fill(mask, -1.0E9)
        attention = torch.softmax(score, dim=-1)
        v = v.view(N, L, self.n_heads, self.out_channels)
        return v*attention

    @property
    def in_channels(self):
        return self.query.in_channels

    @property
    def out_channels(self):
        return self.query.out_channels // self.n_heads

    @property
    def n_heads(self):
        return self._n_heads

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "query": self.query.dump,
                "key": self.key.dump,
                "value": self.value.dump}
