import torch
from typing import Optional
from ._multihead_attention import ATTENTION_TYPE
from ._stages import TransformerEncoderStage, TransformerDecoderStage
from torch.utils.checkpoint import checkpoint


class TransformerEncoder(torch.nn.Module):
    """
    A transformer encoder is a sequence of TransformerEncoderStage
    """

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 RPE_radius: Optional[int] = None, attention_type: ATTENTION_TYPE = "scaled dot product",
                 low_memory: bool = True):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        self.low_memory = low_memory
        for stage in range(n_stages):
            self.stages.append(TransformerEncoderStage(projection_dim, n_heads,
                                                       dropout=dropout, activation=activation,
                                                       RPE_radius=RPE_radius,
                                                       attention_type=attention_type))

    def forward(self, X, padding_mask: Optional[torch.Tensor] = None):
        for stage in self.stages:
            if self.low_memory and self.training:
                X = checkpoint(stage, X, padding_mask)
            else:
                X = stage(X, padding_mask)
        return X


class TransformerDecoder(torch.nn.Module):
    """
    A transformer decoder is a sequence of TransformerDecoderStage
    """

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 RPE_radius: Optional[int] = None, attention_type: ATTENTION_TYPE = "scaled dot product",
                 low_memory: bool = True):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        self.low_memory = low_memory
        for stage in range(n_stages):
            self.stages.append(TransformerDecoderStage(projection_dim, n_heads,
                                                       dropout=dropout, activation=activation,
                                                       RPE_radius=RPE_radius,
                                                       attention_type=attention_type))

    def forward(self, encoded, Y, encoded_padding_mask: Optional[torch.Tensor] = None):
        for stage in self.stages:
            if self.low_memory and self.training:
                Y = checkpoint(stage, encoded, Y, encoded_padding_mask)
            else:
                Y = stage(encoded, Y, encoded_padding_mask)
        return Y
