from pygmalion.neural_networks.layers import MultiHeadAttention
from pygmalion.neural_networks.layers._functional import mask_chronological
import torch
from timeit import timeit
import IPython


masked = True
long_sentence = True
requires_grad = True
device = torch.device("cpu")
# vector dimensions
R = 10
N = 4
H = 5
if long_sentence:
    Lq, Lk, D = 500, 500, 16
else:
    Lq, Lk, D = 20, 20, 512
# Lq, Lk, D, R = 6, 8, 1, 10
# vectors
q = torch.rand((N, H, Lq, D), device=device, requires_grad=requires_grad)
v = torch.rand((N, H, Lk, D), device=device, requires_grad=requires_grad)
# k = torch.rand((N, H, Lk, D), device=device, requires_grad=requires_grad)
RPE = torch.rand((2*R+1, D), device=device, requires_grad=requires_grad)


def naive():
    return MultiHeadAttention._naive_RPE(None, q, RPE, v, masked=masked)


def linear():
    return MultiHeadAttention._linear_RPE(None, q, RPE, v, masked=masked)


for function in [naive, linear]:
    print(f"{function.__name__}:\t", timeit(function, number=10))
    torch.cuda.empty_cache()

res = naive() - linear()

IPython.embed()
