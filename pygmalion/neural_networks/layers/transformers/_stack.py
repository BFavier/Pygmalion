import torch
from itertools import repeat
from typing import Optional, Tuple, Sequence
from .multihead_attention import ATTENTION_TYPE, ScaledDotProductAttention
from ._stages import TransformerEncoderStage, TransformerDecoderStage
from torch.utils.checkpoint import checkpoint


class TransformerEncoder(torch.nn.Module):
    """
    A transformer encoder is a sequence of TransformerEncoderStage
    """

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 gradient_checkpointing: bool = True,
                 attention_type: ATTENTION_TYPE = ScaledDotProductAttention,
                 mask_future: bool=False, expanding_factor: float = 4.0,
                 **kwargs):
        super().__init__()
        self.stages: Sequence[TransformerEncoderStage] = torch.nn.ModuleList()
        self.gradient_checkpointing = gradient_checkpointing
        for stage in range(n_stages):
            self.stages.append(TransformerEncoderStage(projection_dim, n_heads,
                                                       dropout=dropout, activation=activation,
                                                       attention_type=attention_type, mask_future=mask_future,
                                                       expanding_factor=expanding_factor, **kwargs))

    def forward(self, X: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                histories: Optional[Tuple[dict]] = None, attention_kwargs: dict = {}):
        """
        Parameter
        ---------
        X : torch.Tensor
            Tensor of shape (N, L, D) with
            * N sentences count
            * L sequence length
            * D number of features
        padding_mask : torch.tensor or None
            tensor of booleans of shape (N, L) of tokens to ignore
        histories : tuple of dict, or None
            history for each stage
        attention_kwargs : dict
            additional kwargs passed to self attention

        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, D)
        """
        if histories is None:
            histories = repeat(None)
        else:
            assert len(histories) == len(self.stages)
        for history, stage in zip(histories, self.stages):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                X = checkpoint(stage, X, padding_mask, history, attention_kwargs)
            else:
                X = stage(X, padding_mask, history, attention_kwargs)
        return X


class TransformerDecoder(torch.nn.Module):
    """
    A transformer decoder is a sequence of TransformerDecoderStage
    """

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 gradient_checkpointing: bool = True, 
                 attention_type: ATTENTION_TYPE = ScaledDotProductAttention,
                 mask_future: bool=True, expanding_factor: float = 4.0,
                 **kwargs):
        super().__init__()
        self.stages: Sequence[TransformerDecoderStage] = torch.nn.ModuleList()
        self.gradient_checkpointing = gradient_checkpointing
        for stage in range(n_stages):
            self.stages.append(TransformerDecoderStage(projection_dim, n_heads,
                                                       dropout=dropout, activation=activation,
                                                       attention_type=attention_type,
                                                       mask_future=mask_future,
                                                       expanding_factor=expanding_factor, **kwargs))

    def forward(self, Y: torch.Tensor, encoded: torch.Tensor,
                Y_padding_mask : Optional[torch.Tensor] = None,
                encoded_padding_mask: Optional[torch.Tensor] = None,
                histories: Optional[Tuple[dict]] = None,
                self_attention_kwargs: dict = {},
                cross_attention_kwargs: dict = {}):
        """
        Parameter
        ---------
        Y : torch.Tensor
            Tensor of shape (N, Lq, D)
        encoded : torch.Tensor
            Tensor of shape (N, Lk, D)
        Y_padding_mask : torch.tensor or None
            mask of shape (N, Lq)
        encoded_padding_mask : torch.Tensor or None
            mask of shape (N, Lk)
        histories : tuple of dict, or None
            history for each stage
        self_attention_kwargs : dict
            additional kwargs passed to self attention
        cross_attention_kwargs : dict
            additional kwargs passed to cross attention

        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, D)
        """
        if histories is None:
            histories = repeat(None)
        else:
            assert len(histories) == len(self.stages)
        for history, stage in zip(histories, self.stages):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                Y = checkpoint(stage, Y, encoded, Y_padding_mask, encoded_padding_mask, history, self_attention_kwargs, cross_attention_kwargs)
            else:
                Y = stage(Y, encoded, Y_padding_mask, encoded_padding_mask, history, self_attention_kwargs, cross_attention_kwargs)
        return Y
