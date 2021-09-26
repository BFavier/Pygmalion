from pygmalion.neural_networks.layers import MultiHeadAttention
from pygmalion.neural_networks.layers._functional import mask_chronological
import torch
import pandas as pd
import matplotlib.pyplot as plt
from timeit import timeit
import IPython

dot = []
K = []
K_RPE = []

n_rep = 10
N, H = 1, 1
D = 256
L = [2**p for p in range(5, 14)]
# L = [2**p for p in range(13, 14)]
R = 10
m = MultiHeadAttention(D, H, True)
requires_grad = True
device = torch.device("cpu")


def dot_product_attention(q, k, v):
    return m._scaled_dot_product_attention(q, k, v, None)


def kernelized_attention(q, k, v):
    return m._scaleformer_multihead_attention(q, k, v, None, False)


def kernerlized_RPE_attention(q, k, v, RPE):
    return m._scaleformer_multihead_attention(q, k, v, RPE, False)


for l in L:
    print(l)
    Lq, Lk = l, l
    # vectors
    q = torch.rand((N, H, Lq, D), device=device, requires_grad=requires_grad)
    k = torch.rand((N, H, Lk, D), device=device, requires_grad=requires_grad)
    v = torch.rand((N, H, Lk, D), device=device, requires_grad=requires_grad)
    RPE = torch.rand((2*R+1, D), device=device, requires_grad=requires_grad)
    # attention functions
    _dot = lambda: dot_product_attention(q, k, v)
    dot.append(timeit(_dot, number=n_rep))
    _K = lambda: kernelized_attention(q, k, v)
    K.append(timeit(_K, number=n_rep))
    _K_RPE = lambda: kernerlized_RPE_attention(q, k, v, RPE)
    K_RPE.append(timeit(_K_RPE, number=n_rep))


df = pd.DataFrame(data=zip(L, dot, K, K_RPE),
                  columns=["sequences length", "scaled dot product", "kernelized", "kernelized and RPE"])
df.to_csv(r"C:\Users\Benoit\Desktop\MHA_timing.csv", index=False, encoding="utf-8")


plt.style.use("bmh")
f, ax = plt.subplots()
ax.set_title(f"multi head attentions (bidirectional) runtime for d={D} (best of {n_rep})")
ax.set_xscale("log", basex=2)
ax.set_yscale("log", basey=2)
ax.set_xlabel("Sequences length")
ax.set_ylabel("runtime (in seconds)")
ax.plot(L, dot, color="C1", linestyle="--", label="scaled dot product attention")
ax.plot(L, K, linestyle="--", color="C2", label="kernelized attention")
ax.plot(L, K_RPE, linestyle="--", color="C3", label="kernelized attention with RPE")
f.tight_layout()
plt.legend()
plt.show()

IPython.embed()
