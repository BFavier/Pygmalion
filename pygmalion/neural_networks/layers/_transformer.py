import torch
from typing import Optional
from ._activation import Activation
from ._dropout import Dropout
from ._weighting import Linear
from ._batch_norm import BatchNorm0d
from ._multi_head_attention import MultiHeadAttention as MHA


class TransformerEncoderStage(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'TransformerEncoderStage':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.activation = Activation.from_dump(dump["activation"])
        obj.self_attention = MHA.from_dump(dump["self attention"])
        obj.intermediate_norm = BatchNorm0d.from_dump(
            dump["intermediate norm"])
        obj.intermediate_dropout = Dropout.from_dump(
            dump["intermediate dropout"])
        obj.expand = Linear.from_dump(dump["expand"])
        obj.contract = Linear.from_dump(dump["contract"])
        obj.out_dropout = Dropout.from_dump(dump["out dropout"])
        obj.out_norm = BatchNorm0d.from_dump(dump["out norm"])
        return obj

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = Activation(activation)
        self.self_attention = MHA(projection_dim, n_heads)
        self.intermediate_norm = BatchNorm0d(dim)
        self.intermediate_dropout = Dropout(dropout)
        self.expand = Linear(dim, dim*2)
        self.contract = Linear(dim*2, dim)
        self.out_dropout = Dropout(dropout)
        self.out_norm = BatchNorm0d(dim)

    def forward(self, X, mask: Optional[torch.Tensor] = None):
        """
        Parameter
        ---------
        X : torch.Tensor
            Tensor of shape (N, L, F) with
            * N sentences count
            * L sequence length
            * F number of features
        mask : torch.Tensor or None
            mask to apply for the attention

        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, F)
        """
        N, L, _ = X.shape
        input = X.reshape(N*L, -1)
        X = self.self_attention(X, X, mask=mask).reshape(N*L, -1)
        X = self.intermediate_norm(self.intermediate_dropout(X) + input)
        input = X
        X = self.activation(self.expand(X))
        X = self.out_dropout(self.contract(X))
        X = self.out_norm(X + input)
        return X.view(N, L, -1)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "activation": self.activation.dump,
                "self attention": self.self_attention.dump,
                "intermediate norm": self.intermediate_norm.dump,
                "intermediate_dropout": self.intermediate_dropout.dump,
                "expand": self.expand.dump,
                "contract": self.contract.dump,
                "out_dropout": self.out_dropout.dump,
                "out_norm": self.out_norm.dump}


class TransformerDecoderStage(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump: dict) -> 'TransformerEncoderStage':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.activation = Activation.from_dump(dump["activation"])
        obj.masked_attention = MHA.from_dump(dump["masked attention"])
        obj.first_dropout = Dropout.from_dump(dump["first dropout"])
        obj.first_norm = BatchNorm0d.from_dump(dump["first norm"])
        obj.attention = MHA.from_dump(dump["attention"])
        obj.second_dropout = Dropout.from_dump(dump["second dropout"])
        obj.second_norm = BatchNorm0d.from_dump(dump["second norm"])
        obj.expand = Linear.from_dump(dump["expand"])
        obj.contract = Linear.from_dump(dump["contract"])
        obj.out_dropout = Dropout.from_dump(dump["out dropout"])
        obj.out_norm = BatchNorm0d.from_dump(dump["out norm"])
        return obj

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = Activation(activation)
        self.masked_attention = MHA(projection_dim, n_heads)
        self.first_dropout = Dropout(dropout)
        self.first_norm = BatchNorm0d(dim)
        self.attention = MHA(projection_dim, n_heads)
        self.second_dropout = Dropout(dropout)
        self.second_norm = BatchNorm0d(dim)
        self.expand = Linear(dim, 2*dim)
        self.contract = Linear(2*dim, dim)
        self.out_dropout = Dropout(dropout)
        self.out_norm = BatchNorm0d(dim)

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None):
        """
        Parameter
        ---------
        encoded : torch.Tensor
            Tensor of shape (N, L, F)
        Y : torch.Tensor
            Tensor of shape (N, L, F)
        mask : torch.Tensor or None
            mask to apply for the attention

        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, F)
        """
        N, L, _ = Y.shape
        input = Y.reshape(N*L, -1)
        Y = self.masked_attention(Y, Y, mask=mask).reshape(N*L, -1)
        Y = self.first_norm(self.first_dropout(Y) + input).reshape(N, L, -1)
        input = Y.reshape(N*L, -1)
        Y = self.attention(Y, encoded, mask=None).reshape(N*L, -1)
        Y = self.second_norm(self.second_dropout(Y) + input)
        input = Y
        Y = self.out_dropout(self.contract(self.activation(self.expand(Y))))
        Y = self.out_norm(Y + input)
        return Y.view(N, L, -1)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "activation": self.activation.dump,
                "self attention": self.masked_attention.dump,
                "first dropout": self.first_dropout.dump,
                "first norm": self.first_norm,
                "masked attention": self.attention.dump,
                "second dropout": self.second_dropout.dump,
                "second norm": self.second_norm.dump,
                "expand": self.expand.dump,
                "contract": self.contract.dump,
                "out dropout": self.out_dropout.dump,
                "out norm": self.out_norm.dump}

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
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for stage in range(n_stages):
            self.stages.append(TransformerEncoderStage(projection_dim, n_heads,
                               dropout=dropout, activation=activation))

    def forward(self, X, mask=None):
        for stage in self.stages:
            X = stage(X, mask=mask)
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
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for stage in range(n_stages):
            self.stages.append(TransformerDecoderStage(projection_dim, n_heads,
                               dropout=dropout, activation=activation))

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None):
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
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        self.encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout,
                                          activation=activation)
        self.decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout,
                                          activation=activation)

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
