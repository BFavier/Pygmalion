from pygmalion.neural_networks.layers.transformers._functions import _kernelized_attention_linear, _kernelized_attention_naive, _mask_chronological
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from timeit import timeit
import IPython


def kernel(x):
    return F.elu(x) + 1


def naive_m(q, k, v, RPE, scaled=True):
    return _kernelized_attention_naive(kernel, q, k, v, _mask_chronological(Lq, Lk, q.device), None, RPE, scaled=scaled)


def naive_b(q, k, v, RPE, scaled=True):
    _, _, Lq, _ = q.shape
    _, _, Lk, _ = k.shape
    return _kernelized_attention_naive(kernel, q, k, v, None, None, RPE, scaled=scaled)


def linear_m(q, k, v, RPE, scaled=True):
    return _kernelized_attention_linear(kernel, q, k, v, True, None, RPE, scaled=scaled)


def linear_b(q, k, v, RPE, scaled=True):
    _, _, Lq, _ = q.shape
    _, _, Lk, _ = k.shape
    return _kernelized_attention_linear(kernel, q, k, v, False, None, RPE, scaled=scaled)


def test_equality():
    N, H, Lq, Lk, D = 1, 1, 110, 100, 64
    q = torch.rand(N, H, Lq, D)
    k = torch.rand(N, H, Lk, D)
    v = torch.rand(N, H, Lk, D)
    RPE = None
    assert torch.allclose(naive_b(q, k, v, RPE), linear_b(q, k, v, RPE))

def test_equality_masked():
    N, H, Lq, Lk, D = 1, 1, 100, 110, 64
    q = torch.rand(N, H, Lq, D)
    k = torch.rand(N, H, Lk, D)
    v = torch.rand(N, H, Lk, D)
    RPE = None
    assert torch.allclose(naive_m(q, k, v, RPE), linear_m(q, k, v, RPE))

def test_equality_RPE():
    N, H, Lq, Lk, D = 1, 1, 110, 100, 64
    q = torch.rand(N, H, Lq, D)
    k = torch.rand(N, H, Lk, D)
    v = torch.rand(N, H, Lk, D)
    RPE = torch.nn.Embedding(5, D)
    assert torch.allclose(naive_b(q, k, v, RPE), linear_b(q, k, v, RPE))

def benchmark():
    naive_masked = []
    naive_bidirectional = []
    linear_masked = []
    linear_bidirectional = []

    n_rep = 10
    N, H = 1, 1
    D = 64
    L = [2**p for p in range(4, 12)]
    R = 10
    requires_grad = True
    device = torch.device("cpu")
    for l in L:
        print(l)
        Lq, Lk = l, l
        # vectors
        q = torch.rand((N, H, Lq, D), device=device, requires_grad=requires_grad)
        v = torch.rand((N, H, Lk, D), device=device, requires_grad=requires_grad)
        k = torch.rand((N, H, Lk, D), device=device, requires_grad=requires_grad)
        # attention functions
        _naive_m = lambda: naive_m(q, k, v)
        naive_masked.append(timeit(_naive_m, number=n_rep))
        _naive_b = lambda: naive_b(q, k, v)
        naive_bidirectional.append(timeit(_naive_b, number=n_rep))
        _linear_m = lambda: linear_m(q, k, v)
        linear_masked.append(timeit(_linear_m, number=n_rep))
        _linear_b = lambda: linear_b(q, k, v)
        linear_bidirectional.append(timeit(_linear_b, number=n_rep))


    df = pd.DataFrame(data=zip(L, naive_masked, naive_bidirectional, linear_masked, linear_bidirectional),
                    columns=["sequences length", "naive masked", "naive bidirectional", "linear masked", "linear bidirectional"])
    df.to_csv(r"C:\Users\Benoit\Desktop\KA_timing.csv", index=False, encoding="utf-8")


    plt.style.use("bmh")
    f, ax = plt.subplots()
    ax.set_title(f"kernerlized attention runtime for d={D} (best of {n_rep})")
    ax.set_xscale("log", basex=2)
    ax.set_yscale("log", basey=2)
    ax.set_xlabel("Sequences length")
    ax.set_ylabel("runtime (in seconds)")
    ax.plot(L, naive_masked, color="C1", linestyle="-", label="naive masked")
    ax.plot(L, naive_bidirectional, linestyle="--", color="C1", label="naive bidirectional")
    ax.plot(L, linear_masked, linestyle="-", color="C2", label="linear masked")
    ax.plot(L, linear_bidirectional, linestyle="--", color="C2", label="linear bidirectional")
    f.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    N, H, Lq, Lk, D = 1, 1, 110, 100, 64
    q = torch.rand(N, H, Lq, D)
    k = torch.rand(N, H, Lk, D)
    v = torch.rand(N, H, Lk, D)
    test_equality_RPE()
    IPython.embed()