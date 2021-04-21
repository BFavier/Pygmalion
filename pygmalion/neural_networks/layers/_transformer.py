import torch
from typing import List
from ._dense import Dense0d
from ._weighting import Linear
from ._batch_norm import BatchNorm0d
from ._multi_head_attention import MultiHeadAttention as MHA


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
        self.output = Linear(self.dense.out_channels(in_features),
                             self.out_features)
        self.norm = BatchNorm0d

    def forward(self, X):
        input = X
        N, L, _ = X.shape
        X = self.self_attention(X, X, masked=False)
        X = self.dense(X.view(N*L, -1))
        X = self.output(X) + input.view(N*L, -1)
        return X.view(N, L, -1)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "self attention": self.self_attention.dump,
                "dense": self.dense.dump,
                "output": self.output.dump}

    @property
    def in_features(self):
        return self.projection_dim * self.n_heads

    @property
    def out_features(self):
        return self.projection_dim * self.n_heads


class TransformerDecoderStage(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'TransformerEncoderStage':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.self_attention = MHA.from_dump(dump["self attention"])
        obj.attention = MHA.from_dump(dump["attention"])
        obj.dense = Dense0d.from_dump(dump["dense"])
        obj.output = Linear.from_dump(dump["output"])
        return obj

    def __init__(self, projection_dim: int, n_heads: int,
                 hidden_layers: List[dict], out_features: int, **kwargs):
        super().__init__()
        self.self_attention = MHA(projection_dim, n_heads)
        in_features = projection_dim * n_heads
        self.attention = MHA(projection_dim, n_heads)
        self.dense = Dense0d(in_features, hidden_layers, **kwargs)
        self.output = Linear(self.dense.out_channels(in_features),
                             out_features)

    def forward(self, encoded, Y):
        input = Y
        N, L, _ = Y.shape
        Y = self.self_attention(Y, Y, masked=True)
        Y = self.attention(Y, encoded, masked=False)
        Y = self.dense(Y.view(N*L, -1))
        Y = self.output(Y) + input.view(N*L, -1)
        return Y.view(N, L, -1)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "self attention": self.self_attention.dump,
                "attention": self.attention.dump,
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
        for stage in n_stages:
            self.stages.append(TransformerEncoderStage(projection_dim, n_heads,
                                                       hidden_layers, **kwargs)
                               )

    def forward(self, X):
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
        for stage in n_stages:
            self.stages.append(TransformerDecoderStage(projection_dim, n_heads,
                                                       hidden_layers, **kwargs)
                               )

    def forward(self, encoded, Y):
        for stage in self.stages:
            Y = stage(encoded, Y)
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
        obj.decoder = TransformerDecoder.from_dump(dump["decoder"])

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 hidden_layers: List[dict], **kwargs):
        self.encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                          hidden_layers, **kwargs)
        self.decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                          hidden_layers, **kwargs)

    def forward(self, X):
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
        X = self._positional_encoding(X)
        return self.encoder(X)

    def decode(self, encoded, Y):
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
        return self.decoder(encoded, Y)

    def _positional_encoding(self, X):
        """
        Performs positional encoding on the input, in the
        "Attention is all you need" paper fashion.

        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (..., D), with D the embedding dimension

        Returns
        -------
        torch.Tensor:
            tensor of shape (..., D)
        """
        shape = X.shape
        X = X.view(-1, shape[-1])
        N, D = X.shape
        pe = torch.zeros(N, D, dtype=torch.float, device=X.device)
        position = torch.arange(0, D, dtype=torch.float).unsqueeze(0)
        angle = position / 10000**(2*(position//2)/D)
        pe[:, 0::2] = torch.cos(angle[:, 0::2])
        pe[:, 1::2] = torch.sin(angle[:, 1::2])
        X = (X + pe).view(shape)
        return X

    @property
    def dump(self) -> dict:
        pass
