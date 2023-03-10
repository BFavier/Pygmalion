import torch
from typing import Optional, Tuple, Literal, Callable
from ._attention import _scaled_dot_product_attention, _kernelized_attention_linear, _kernelized_attention_naive
from ._utilities import _log_exp_kernel
from types import LambdaType

ATTENTION_TYPE = Literal["scaled dot product", "kernelized linear", "kernelized quadratic"]
_attention_functions = {k: v for k, v in zip(ATTENTION_TYPE, (_scaled_dot_product_attention, _kernelized_attention_linear, _kernelized_attention_naive))}


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 masked: bool,
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
        self.relative_positional_encoding = torch.nn.Embedding(2*RPE_radius+1, dim) if RPE_radius else None
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)
        self.attention = _attention_functions[attention_type]
        assert not isinstance(kernel_function, LambdaType), "Lambda function cannot be pickled and saved on disk"
        self.kernel_function = kernel_function if "kernelized" in attention_type else None

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                query_padding_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
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
        key_padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, Lk)
            or None if padding tokens should not be masked
        query_padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, Lq)
            or None if padding tokens should not be masked
        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Lq, D)
        """
        query, key = query.to(self.device), key.to(self.device)
        N, Lq, _ = query.shape
        N, Lk, _ = key.shape
        # project into 'n_heads' different subspaces
        q = self.query(query).reshape(N, Lq, self.n_heads, self.projection_dim)
        k = self.key(key).reshape(N, Lk, self.n_heads, self.projection_dim)
        v = self.value(key).reshape(N, Lk, self.n_heads, self.projection_dim)
        # compute attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.kernel_function is None:
            attention = self.attention(q, k, v, self.masked, key_padding_mask)
        else:
            attention = self.attention(self.kernel_function, q, k, v, self.masked, key_padding_mask)
        attention = attention.transpose(2, 1).reshape(N, Lq, -1)
        if query_padding_mask is not None:
            query_padding_mask = query_padding_mask.to(attention.device).unsqueeze(-1)
            attention = torch.masked_fill(attention, query_padding_mask, 0.)
        return attention

    @property
    def device(self) -> torch.device:
        return self.query.weight.device