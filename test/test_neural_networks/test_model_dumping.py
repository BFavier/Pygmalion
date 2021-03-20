import sys
import torch
import pygmalion.neural_networks as nn

rtol, atol = 1.0E-4, 0

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
