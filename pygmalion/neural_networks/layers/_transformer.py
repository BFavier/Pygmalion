import torch
from typing import List, Optional
from ._dense import Dense0d
from ._weighting import Linear
from ._multi_head_attention import MultiHeadAttention as MHA
from ._functional import positional_encoding


class TransformerEncoderStage(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'TransformerEncoderStage':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.self_attention = MHA.from_dump(dump["self attention"])
        obj.dense = Dense0d.from_dump(dump["dense"])
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self, projection_dim: int, n_heads: int,
                 hidden_layers: List[dict], **kwargs):
        super().__init__()
        self.self_attention = MHA(projection_dim, n_heads)
        in_features = projection_dim * n_heads
        self.dense = Dense0d(in_features, hidden_layers, **kwargs)
        self.output = Linear(self.dense.out_features(in_features),
                             in_features)

    def forward(self, X):
        input = X
        N, L, _ = X.shape
        X = self.self_attention(X, X)
        X = self.dense(X.view(N*L, -1))
        X = self.output(X) + input.view(N*L, -1)
        return X.view(N, L, -1)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "self attention": self.self_attention.dump,
                "dense": self.dense.dump,
                "output": self.output.dump}


class TransformerDecoderStage(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'TransformerEncoderStage':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.self_attention = MHA.from_dump(dump["self attention"])
        obj.masked_attention = MHA.from_dump(dump["attention"])
        obj.dense = Dense0d.from_dump(dump["dense"])
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self, projection_dim: int, n_heads: int,
                 hidden_layers: List[dict], **kwargs):
        super().__init__()
        self.self_attention = MHA(projection_dim, n_heads)
        in_features = projection_dim * n_heads
        self.masked_attention = MHA(projection_dim, n_heads)
        self.dense = Dense0d(in_features, hidden_layers, **kwargs)
        self.output = Linear(self.dense.out_features(in_features),
                             in_features)

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None):
        input = Y
        N, L, _ = Y.shape
        Y = self.self_attention(Y, Y, mask=None)
        Y = self.masked_attention(Y, encoded, mask=mask)
        Y = self.dense(Y.view(N*L, -1))
        Y = self.output(Y) + input.view(N*L, -1)
        return Y.view(N, L, -1)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "self attention": self.self_attention.dump,
                "masked attention": self.masked_attention.dump,
                "dense": self.dense.dump,
                "output": self.output.dump}

    @property
    def in_features(self):
        return self.projection_dim * self.n_heads

    @property
    def out_features(self):
        return self.projection_dim * self.n_heads


class TransformerEncoder(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'TransformerEncoder':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.stages = torch.nn.ModuleList()
        for stage in dump["stages"]:
            obj.stages.append(TransformerEncoderStage.from_dump(stage))
        return obj

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 hidden_layers: List[dict], **kwargs):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for stage in range(n_stages):
            self.stages.append(TransformerEncoderStage(projection_dim, n_heads,
                                                       hidden_layers, **kwargs)
                               )

    def forward(self, X):
        X = positional_encoding(X)
        for stage in self.stages:
            X = stage(X)
        return X

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "stages": [stage.dump for stage in self.stages]}


class TransformerDecoder(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'TransformerDecoder':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.stages = torch.nn.ModuleList()
        for stage in dump["stages"]:
            obj.stages.append(TransformerDecoderStage.from_dump(stage))
        return obj

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 hidden_layers: List[dict], **kwargs):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for stage in range(n_stages):
            self.stages.append(TransformerDecoderStage(projection_dim, n_heads,
                                                       hidden_layers, **kwargs)
                               )

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None):
        Y = positional_encoding(Y)
        for stage in self.stages:
            Y = stage(encoded, Y, mask=mask)
        return Y

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "stages": [stage.dump for stage in self.stages]}


class Transformer(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'Transformer':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.encoder = TransformerEncoder.from_dump(dump["encoder"])
        obj.decoder = TransformerDecoder.from_dump(dump["encoder"])
        return obj

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 hidden_layers: List[dict], **kwargs):
        super().__init__()
        self.encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                          hidden_layers, **kwargs)
        self.decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                          hidden_layers, **kwargs)

    def forward(self, X):
        return self.encode(X)

    def encode(self, X):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            tensor of embedded input tokens
            tensor of floats of shape (N, L, D) with:
            * N : number of sentences
            * L : tokens per sentence
            * D : the embedding dimension

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, L, D)
        """
        return self.encoder(X)

    def decode(self, encoded, Y, mask: Optional[torch.Tensor] = None):
        """
        performs the decoding part of the network:
        for each of the already predicted tokens, predict the next token.

        Parameters
        ----------
        encoded : torch.Tensor
            tensor of encoded inputs
            tensor of floats of shape (N, Lx, D) with:
            * N : number of sentences
            * Lx : tokens per sentence in the input language
            * D : embedding dim

        Y : torch.Tensor
            tensor of the already predicted tokens
            tensor of long of shape (N, Ly, D) with:
            * N : number of sentences
            * Ly : tokens per sentence in the output language
            * D : embedding dim

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, Ly, D)
        """
        Y = self.decoder(encoded, Y, mask=mask)
        return Y

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "encoder": self.encoder.dump,
                "decoder": self.decoder.dump}
