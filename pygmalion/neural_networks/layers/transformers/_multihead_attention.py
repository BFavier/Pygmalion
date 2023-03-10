import torch
from typing import Optional, Tuple, Literal, Callable
from ._attention import _scaled_dot_product_attention, _kernelized_attention_linear, _kernelized_attention_naive
from ._utilities import _log_exp_kernel
from types import LambdaType

ATTENTION_TYPE = Literal["scaled dot product", "kernelized linear", "kernelized quadratic"]
_attention_functions = {k: v for k, v in zip(ATTENTION_TYPE, (_scaled_dot_product_attention, _kernelized_attention_linear, _kernelized_attention_naive))}


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 masked: bool, key_padding_mask: Optional[torch.Tensor] = None,
                 query_padding_mask: Optional[torch.Tensor] = None,
                 RPE_radius: Optional[int] = None,
                 attention_type: ATTENTION_TYPE = "scaled dot product",
                 kernel_function : Callable = _log_exp_kernel):
        f"""
        Parameters
        ----------
        projection_dim : int
            the dimension of the projection space for the feature vectors
        n_heads : int
            the number of different projection at each stage of the transformer
        masked: bool
            whether or not a query at index i can't attend to keys
            at index j > i in the sequence
        key_padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, Lk)
            or None if padding tokens should not be masked
        query_padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, Lq)
            or None if padding tokens should not be masked
        RPE_radius : int or None
            The radius of the relative positional encoding
            or None if no relative positional encoding should be applied
        attention_type : {ATTENTION_TYPE}
            the type of attention function to perform
        kernel_function : Callable
            the kernel function for kernelized attention
        """
        super().__init__()
        self.n_heads = n_heads
        self.projection_dim = projection_dim
        dim = projection_dim * n_heads
        self.masked = masked
        self.key_padding_mask = key_padding_mask
        self.query_padding_mask = query_padding_mask
        self.relative_positional_encoding = torch.nn.Embedding(2*RPE_radius+1, dim) if RPE_radius else None
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)
        self.attention = _attention_functions[attention_type]
        assert not isinstance(kernel_function, LambdaType), "Lambda function cannot be pickled and saved on disk"
        self.kernel_function = kernel_function

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
