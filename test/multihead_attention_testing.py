from pygmalion.neural_networks.layers import MultiHeadAttention
from pygmalion.neural_networks.layers._functional import mask_chronological
import torch
from timeit import timeit
import IPython

masked = False
long_sentence = True
requires_grad = True
device = torch.device("cpu")
# vector dimensions
R = 10
N = 10
H = 8
if long_sentence:
    Lq, Lk, D = 500, 500, 8
else:
    Lq, Lk, D = 20, 20, 512
# vectors
q = torch.rand((N, H, Lq, D), device=device, requires_grad=requires_grad)
v = torch.rand((N, H, Lk, D), device=device, requires_grad=requires_grad)
k = torch.rand((N, H, Lk, D), device=device, requires_grad=requires_grad)
RPE = torch.rand((2*R+1, D), device=device, requires_grad=requires_grad)
mask = mask_chronological(Lq, Lk, device) if masked else None
# attention functions


def original():
    return MultiHeadAttention._scaled_dot_product_attention(None, q, k, v,
                                                            mask)[0]


def linear():
    return MultiHeadAttention._linear_complexity_attention(None, q, k, v,
                                                           masked)[0]


for function in [original, linear]:
    print(f"{function.__name__}:  \t", timeit(function, number=10))
    torch.cuda.empty_cache()


IPython.embed()
