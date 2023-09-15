from typing import Optional
from pygmalion.neural_networks.layers.transformers.multihead_attention import FourrierKernelAttention
import torch
import torch.nn.functional as F
# import pandas as pd
# import matplotlib.pyplot as plt
# from timeit import timeit


def kernel(x):
    return F.elu(x) + 1


def naive_m(q, k, v, c, pq, pk, key_mask: Optional[torch.Tensor]=None, scaled: bool=True):
    return FourrierKernelAttention._attention_naive(q, k, v, c, pq, pk, mask_future=True,
                                                    key_mask=key_mask, scaled=scaled, future_offset=0)


def naive_b(q, k, v, c, pq, pk, key_mask: Optional[torch.Tensor]=None, scaled: bool=True):
    return FourrierKernelAttention._attention_naive(q, k, v, c, pq, pk, mask_future=False,
                                                    key_mask=key_mask, scaled=scaled, future_offset=0)


def linear_m(q, k, v, c, pq, pk, key_mask: Optional[torch.Tensor]=None, scaled: bool=True):
    return FourrierKernelAttention._attention_linear(q, k, v, c, pq, pk, mask_future=True,
                                                    key_mask=key_mask, scaled=scaled, future_offset=0)


def linear_b(q, k, v, c, pq, pk, key_mask: Optional[torch.Tensor]=None, scaled: bool=True):
    return FourrierKernelAttention._attention_linear(q, k, v, c, pq, pk, mask_future=False,
                                                    key_mask=key_mask, scaled=scaled, future_offset=0)


def test_equality_bidirectional():
    N, H, Lq, Lk, D = 1, 1, 110, 100, 64
    q = kernel(torch.rand(N, H, Lq, D))
    k = kernel(torch.rand(N, H, Lk, D))
    v = torch.rand(N, H, Lk, D)
    c = torch.rand(H, D)
    pq = torch.rand(N, H, Lq, D)
    pk = torch.rand(N, H, Lk, D)
    assert torch.allclose(naive_b(q, k, v, c, pq, pk), linear_b(q, k, v, c, pq, pk))

def test_equality_masked():
    N, H, Lq, Lk, D = 1, 1, 100, 110, 64
    q = kernel(torch.rand(N, H, Lq, D))
    k = kernel(torch.rand(N, H, Lk, D))
    v = torch.rand(N, H, Lk, D)
    c = torch.rand(H, D)
    pq = torch.rand(N, H, Lq, D)
    pk = torch.rand(N, H, Lk, D)
    assert torch.allclose(naive_m(q, k, v, c, pq, pk), linear_m(q, k, v, c, pq, pk))


if __name__ == "__main__":
    test_equality_bidirectional()
    test_equality_masked()
    import IPython
    IPython.embed()