import sys
import torch
import pygmalion.neural_networks.layers as lay

rtol, atol = 1.0E-4, 0


def test_Activated():
    for cls, dim in [(lay.Activated0d, 0), (lay.Activated1d, 1),
                     (lay.Activated2d, 2)]:
        for padded in [True, False]:
            for activation in ["relu", "tanh", "leaky_relu", "elu"]:
                for stacked in [True, False]:
                    kwargs = {"activation": activation, "stacked": stacked}
                    if dim > 0:
                        kwargs["padded"] = padded
                    C_in, C_out = 5, 3
                    shape = [10, C_in] + [10]*dim
                    lay1 = cls(C_in, C_out, **kwargs)
                    lay2 = cls.from_dump(lay1.dump)
                    tensor = torch.rand(size=shape, dtype=torch.float)
                    assert torch.allclose(lay1(tensor), lay2(tensor),
                                          rtol=rtol, atol=atol)


def test_BatchNorm():
    for cls, dim in [(lay.BatchNorm0d, 0), (lay.BatchNorm1d, 1),
                     (lay.BatchNorm2d, 2)]:
        for momentum in [0.1, None]:
            for affine in [True, False]:
                kwargs = {"momentum": momentum, "affine": affine}
                C = 5
                shape = [10, C] + [10]*dim
                lay1 = cls(C, **kwargs)
                lay2 = cls.from_dump(lay1.dump)
                tensor = torch.rand(size=shape, dtype=torch.float)
                assert torch.allclose(lay1(tensor), lay2(tensor),
                                      rtol=rtol, atol=atol)


def test_Decoder():
    for cls, dim in [(lay.Decoder1d, 1), (lay.Decoder2d, 2)]:
        dense_layers = [[{"channels": 4}, {"channels": 8}], {"channels": 16}]
        upsampling_factors = [(2, 3)[:dim], (5, 1)[:dim]]
        kwargs = dict()
        C = 5
        shape = [10, C] + [10]*dim
        lay1 = cls(C, dense_layers, upsampling_factors, **kwargs)
        lay2 = cls.from_dump(lay1.dump)
        tensor = torch.rand(size=shape, dtype=torch.float)
        assert torch.allclose(lay1(tensor), lay2(tensor),
                              rtol=rtol, atol=atol)


def test_Dense():
    for cls, dim in [(lay.Dense0d, 0), (lay.Dense1d, 1), (lay.Dense2d, 2)]:
        layers = [{"channels": 8}, {"channels": 8}]
        kwargs = dict()
        C = 5
        shape = [10, C] + [10]*dim
        lay1 = cls(C, layers, **kwargs)
        lay2 = cls.from_dump(lay1.dump)
        tensor = torch.rand(size=shape, dtype=torch.float)
        assert torch.allclose(lay1(tensor), lay2(tensor),
                              rtol=rtol, atol=atol)


def test_Downsampling():
    for cls, dim in [(lay.Downsampling1d, 1), (lay.Downsampling2d, 2)]:
        for pooling_type in ["max", "avg"]:
            dense_layers = [{"channels": 4}, {"channels": 8}, {"channels": 16}]
            pooling_window = (2, 3)[:dim]
            kwargs = dict()
            C = 5
            shape = [10, C] + [10]*dim
            lay1 = cls(C, dense_layers, pooling_window, pooling_type, **kwargs)
            lay2 = cls.from_dump(lay1.dump)
            tensor = torch.rand(size=shape, dtype=torch.float)
            assert torch.allclose(lay1(tensor), lay2(tensor),
                                  rtol=rtol, atol=atol)


def test_Embeding():
    lay1 = lay.Embedding(100, 10)
    lay2 = lay.Embedding.from_dump(lay1.dump)
    tensor = torch.randint(0, 100, size=(100, 20), dtype=torch.long)
    assert torch.allclose(lay1(tensor), lay2(tensor),
                          rtol=rtol, atol=atol)


def test_Encoder():
    for cls, dim in [(lay.Encoder1d, 1), (lay.Encoder2d, 2)]:
        dense_layers = [[{"channels": 4}, {"channels": 8}], {"channels": 16}]
        pooling_windows = [(2, 3)[:dim], (5, 1)[:dim]]
        kwargs = dict()
        C = 5
        shape = [10, C] + [10]*dim
        lay1 = cls(C, dense_layers, pooling_windows, **kwargs)
        lay2 = cls.from_dump(lay1.dump)
        tensor = torch.rand(size=shape, dtype=torch.float)
        assert torch.allclose(lay1(tensor), lay2(tensor),
                              rtol=rtol, atol=atol)


def test_MultiHeadAttention():
    lay1 = lay.MultiHeadAttention(10, 4)
    lay2 = lay.MultiHeadAttention.from_dump(lay1.dump)
    tensor = torch.rand(size=(100, 20, 40), dtype=torch.float)
    assert torch.allclose(lay1(tensor, tensor, masked=True),
                          lay2(tensor, tensor, masked=True),
                          rtol=rtol, atol=atol)


def test_Padding():
    for cls, dim in [(lay.Padding1d, 1), (lay.Padding2d, 2)]:
        padding = (2, 3, 4, 5)[:2*dim]
        value = -1.2
        kwargs = dict()
        C = 5
        shape = [10, C] + [10]*dim
        lay1 = cls(padding, value, **kwargs)
        lay2 = cls.from_dump(lay1.dump)
        tensor = torch.rand(size=shape, dtype=torch.float)
        assert torch.allclose(lay1(tensor), lay2(tensor),
                              rtol=rtol, atol=atol)


def test_Pooling():
    for cls, dim in [(lay.Pooling1d, 1), (lay.Pooling2d, 2)]:
        for window in [(8, 3), None]:
            for pooling_type in ["max", "avg"]:
                if window is not None:
                    window = window[:dim]
                kwargs = dict()
                C = 5
                shape = [10, C] + [10]*dim
                lay1 = cls(window, pooling_type, **kwargs)
                lay2 = cls.from_dump(lay1.dump)
                tensor = torch.rand(size=shape, dtype=torch.float)
                assert torch.allclose(lay1(tensor), lay2(tensor),
                                      rtol=rtol, atol=atol)


def test_UNet():
    for cls, dim in [(lay.UNet1d, 1), (lay.UNet2d, 2)]:
        downsampling = [[{"channels": 4}, {"channels": 8}], {"channels": 16}]
        pooling = [(2, 3)[:dim], (5, 1)[:dim]]
        upsampling = [{"channels": 15}, [{"channels": 10}, {"channels": 5}]]
        kwargs = dict()
        C = 5
        shape = [10, C] + [10]*dim
        lay1 = cls(C, downsampling, pooling, upsampling, **kwargs)
        lay2 = cls.from_dump(lay1.dump)
        tensor = torch.rand(size=shape, dtype=torch.float)
        assert torch.allclose(lay1(tensor), lay2(tensor),
                              rtol=rtol, atol=atol)


def test_Unpooling():
    for cls, dim in [(lay.Unpooling1d, 1), (lay.Unpooling2d, 2)]:
        for method in ["nearest", "interpolate"]:
            if dim == 1:
                factor = 3
            else:
                factor = [2, 3]
            kwargs = dict()
            C = 5
            shape = [10, C] + [10]*dim
            lay1 = cls(factor, method, **kwargs)
            lay2 = cls.from_dump(lay1.dump)
            tensor = torch.rand(size=shape, dtype=torch.float)
            assert torch.allclose(lay1(tensor), lay2(tensor),
                                  rtol=rtol, atol=atol)


def test_Upsampling():
    for cls, dim in [(lay.Upsampling1d, 1), (lay.Upsampling2d, 2)]:
        for upsampling_method in ["nearest", "interpolate"]:
            dense_layers = [{"channels": 4}, {"channels": 8}, {"channels": 16}]
            if dim == 1:
                upsampling_factors = 3
            else:
                upsampling_factors = (2, 3)
            kwargs = dict()
            C = 5
            shape = [10, C] + [10]*dim
            lay1 = cls(C, dense_layers, upsampling_factors, upsampling_method,
                       **kwargs)
            lay2 = cls.from_dump(lay1.dump)
            tensor = torch.rand(size=shape, dtype=torch.float)
            assert torch.allclose(lay1(tensor), lay2(tensor),
                                  rtol=rtol, atol=atol)


def test_Weighting():
    for cls, dim in [(lay.Linear, 0), (lay.Conv1d, 1), (lay.Conv2d, 2)]:
        for bias in [True, False]:
            kwargs = {"bias": bias}
            C_in, C_out = 5, 3
            if dim == 0:
                args = (C_in, C_out)
            elif dim == 1:
                kernel_size = 3
                args = (C_in, C_out, kernel_size)
                kwargs["stride"] = 3
            elif dim == 2:
                kernel_size = (2, 3)
                args = (C_in, C_out, kernel_size)
                kwargs["stride"] = (3, 2)
            shape = [10, C_in] + [10]*dim
            lay1 = cls(*args, **kwargs)
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
