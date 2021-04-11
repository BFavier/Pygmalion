import torch


class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)

    @classmethod
    def from_dump(cls, dump: dict) -> 'Embedding':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.weight = torch.nn.Parameter(torch.tensor(dump["weight"],
                                                     dtype=torch.float))
        obj.padding_idx = None
        obj.max_norm = None
        obj.norm_type = 2.0
        obj.scale_grad_by_freq = False
        obj.sparse = False
        return obj

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "weight": self.weight.tolist()}
