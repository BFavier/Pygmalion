import torch
from typing import Union, Optional, Tuple, List
from torch.utils.checkpoint import checkpoint
import scipy.cluster.hierarchy as scp


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int):
        """
        Parameters
        ----------
        projection_dim : int
            the dimension of the projection space for the feature vectors
        n_heads : int
            the number of different projection at each stage of the transformer
        linear_complexity : bool
            if True, a variant of the attention mechanism is used that has
            linear coplexity with the sequence length of tokens
        """
        super().__init__()
        self.n_heads = n_heads
        self.projection_dim = projection_dim
        dim = projection_dim * n_heads
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None):
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
        padding_mask : torch.Tensor or None
            the padding mask, tensor of booleans of shape (N, Lk),
            where value vectors are set to 0 in the attention function
        null_mask
        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Lq, D)
        """
        return self._multihead_attention(query, key, mask, padding_mask)

    def _multihead_attention(self, query: torch.Tensor, key: torch.Tensor,
                             mask: Optional[torch.Tensor],
                             padding_mask: Optional[torch.Tensor]
                             ) -> torch.Tensor:
        """
        Apply multihead attention.
        Same inputs/outputs types/shapes as the forward pass
        """
        N, Lq, _ = query.shape
        N, Lk, _ = key.shape
        # project into 'n_heads' different subspaces
        q = self.query(query).reshape(N, Lq, self.n_heads, self.projection_dim)
        k = self.key(key).reshape(N, Lk, self.n_heads, self.projection_dim)
        v = self.value(key).reshape(N, Lk, self.n_heads, self.projection_dim)
        # compute attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attention = self._scaled_dot_product_attention(q, k, v, mask, padding_mask)
        attention = attention.transpose(2, 1).reshape(N, Lq, -1)
        return attention

    def _scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor,
                                      v: torch.Tensor,
                                      mask: Optional[torch.Tensor],
                                      padding_mask: Optional[torch.Tensor]
                                      ) -> Tuple[torch.Tensor]:
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
            query tensor of shape (N, H, Lq, D)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, D)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, D)
        mask : torch.Tensor or None
            tensor of booleans of shape (Lq, Lk)
        padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, Lk)
        Returns
        -------
        tuple of torch.Tensors:
            a tuple of (attention, score)
        """
        N, H, Lk, d = k.shape
        scaling = Lk**0.5 if padding_mask is None else (~padding_mask).float().sum(dim=-1).reshape(N, 1, 1, 1)**0.5
        score = torch.matmul(q, k.transpose(-2, -1)) / scaling
        if mask is not None:
            score = score.masked_fill(mask, -float("inf"))
        if padding_mask is not None:
            score = score.masked_fill(padding_mask.reshape(N, 1, 1, Lk), -float("inf"))
        score = torch.softmax(score, dim=-1)
        attention = torch.matmul(score, v)
        return attention

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
        embed_dim = self.projection_dim * self.n_heads
        in_proj_weight = None
        in_proj_bias = None
        out_proj_weight = torch.eye(embed_dim, dtype=torch.float,
                                    device=query.device)
        out_proj_bias = torch.zeros(embed_dim, dtype=torch.float,
                                    device=query.device)
        dropout = 0.
        return torch.nn.functional.multi_head_attention_forward(
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


class TransformerEncoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None,
                 activation: str = "relu"):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = getattr(torch, activation)
        self.self_attention = MultiHeadAttention(projection_dim, n_heads)
        self.intermediate_norm = torch.nn.LayerNorm(dim)
        self.intermediate_dropout = torch.nn.Dropout(dropout)
        self.expand = torch.nn.Linear(dim, dim * 4)
        self.contract = torch.nn.Linear(dim * 4, dim)
        self.out_dropout = torch.nn.Dropout(dropout)
        self.out_norm = torch.nn.LayerNorm(dim)

    def forward(self, X, mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None):
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
        X = X.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        N, L, _ = X.shape
        input = X.reshape(N * L, -1)
        X = self.self_attention(X, X, mask=mask, padding_mask=padding_mask).reshape(N * L, -1)
        X = self.intermediate_dropout(X) + input
        X = self.intermediate_norm(X)
        input = X
        X = self.activation(self.expand(X))
        X = self.out_dropout(self.contract(X)) + input
        X = self.out_norm(X)
        return X.reshape(N, L, -1)

    @property
    def device(self) -> torch.device:
        return self.self_attention.key.weight.device


class TransformerDecoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu"):
        super().__init__()
        dim = projection_dim * n_heads
        self.activation = getattr(torch, activation)
        self.masked_attention = MultiHeadAttention(projection_dim, n_heads)
        self.first_dropout = torch.nn.Dropout(dropout)
        self.first_norm = torch.nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(projection_dim, n_heads)
        self.second_dropout = torch.nn.Dropout(dropout)
        self.second_norm = torch.nn.LayerNorm(dim)
        self.expand = torch.nn.Linear(dim, 4 * dim)
        self.contract = torch.nn.Linear(4 * dim, dim)
        self.out_dropout = torch.nn.Dropout(dropout)
        self.out_norm = torch.nn.LayerNorm(dim)

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None):
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
        encoded = encoded.to(self.device)
        Y = Y.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        N, L, _ = Y.shape
        input = Y.reshape(N * L, -1)
        Y = self.masked_attention(Y, Y, mask=mask, padding_mask=padding_mask).reshape(N * L, -1)
        Y = self.first_norm(self.first_dropout(Y) + input).reshape(N, L, -1)
        input = Y.reshape(N * L, -1)
        Y = self.attention(Y, encoded, mask=None, padding_mask=padding_mask).reshape(N * L, -1)
        Y = self.second_norm(self.second_dropout(Y) + input)
        input = Y
        Y = self.out_dropout(self.contract(self.activation(self.expand(Y))))
        Y = self.out_norm(Y + input)
        return Y.reshape(N, L, -1)

    @property
    def device(self) -> torch.device:
        return self.masked_attention.key.weight.device


class TransformerEncoder(torch.nn.Module):

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 low_memory: bool = True):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        self.low_memory = low_memory
        for stage in range(n_stages):
            self.stages.append(TransformerEncoderStage(projection_dim, n_heads,
                                                       dropout=dropout, activation=activation))

    def forward(self, X, mask=None, padding_mask: Optional[torch.Tensor] = None):
        for stage in self.stages:
            if self.low_memory and self.training:
                X = checkpoint(stage, X, mask, padding_mask)
            else:
                X = stage(X, mask=mask, padding_mask=padding_mask)
        return X


class TransformerDecoder(torch.nn.Module):

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 low_memory: bool = True):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        self.low_memory = low_memory
        for stage in range(n_stages):
            self.stages.append(TransformerDecoderStage(projection_dim, n_heads,
                                                       dropout=dropout, activation=activation))

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None):
        for stage in self.stages:
            if self.low_memory and self.training:
                Y = checkpoint(stage, encoded, Y, mask, padding_mask)
            else:
                Y = stage(encoded, Y, mask=mask, padding_mask=padding_mask)
        return Y
