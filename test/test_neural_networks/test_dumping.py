import sys
import torch
import pandas as pd
import numpy as np
import pygmalion.neural_networks.layers as lay
import pygmalion.neural_networks as nn

rtol, atol = 1.0E-4, 0


def test_Activated():
    for cls, dim in [(lay.Activated0d, 0), (lay.Activated1d, 1),
                     (lay.Activated2d, 2)]:
        for padded in [True, False]:
            for bias in [True, False]:
                for stacked in [True, False]:
                    kwargs = {"bias": bias, "stacked": stacked}
                    if dim > 0:
                        kwargs["padded"] = padded
                    C_in, C_out = 5, 3
                    shape = [10, C_in] + [10]*dim
                    lay1 = cls(C_in, C_out, **kwargs)
                    lay2 = cls.from_dump(lay1.dump)
                    tensor = torch.rand(size=shape, dtype=torch.float)
                    assert torch.allclose(lay1(tensor), lay2(tensor),
                                          rtol=rtol, atol=atol)


if __name__ == "__main__":
    module = sys.modules[__name__]
    for attr in dir(module):
        if not attr.startswith("test_"):
            continue
        print(attr, end="")
        func = getattr(module, attr)
        try:
            func()
        except AssertionError:
            print(": Failed")
        else:
            print(": Passed")
