import sys
import torch
import pygmalion.neural_networks.layers as lay
import pygmalion.neural_networks as nn

rtol, atol = 1.0E-4, 0


def test_ConvStage1d():
    C = 5
    for padded in [True, False]:
        for with_bias in [True, False]:
            for stacked in [True, False]:
                lay1 = lay.ConvStage1d(C, 3, 7, stride=2, padded=padded,
                                       with_bias=with_bias,
                                       stacked=stacked)
                lay2 = lay.ConvStage1d.from_dump(lay1.dump)
                tensor = torch.rand(size=(7, C, 16), dtype=torch.float)
                assert torch.allclose(lay1(tensor), lay2(tensor),
                                      rtol=rtol, atol=atol)


def test_ConvStage2d():
    C = 5
    for padded in [True, False]:
        for with_bias in [True, False]:
            for stacked in [True, False]:
                lay1 = lay.ConvStage2d(C, 3, (7, 3), stride=(3, 2),
                                       padded=padded,
                                       with_bias=with_bias,
                                       stacked=stacked)
                lay2 = lay.ConvStage2d.from_dump(lay1.dump)
                tensor = torch.rand(size=(7, C, 32, 16), dtype=torch.float)
                assert torch.allclose(lay1(tensor), lay2(tensor),
                                      rtol=rtol, atol=atol)


def test_LinearStage():
    C = 6
    lay1 = lay.LinearStage(C, 5)
    lay2 = lay.LinearStage.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, C), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_Activation():
    for name in ["relu", "leaky_relu", "tanh", "sigmoid"]:
        lay1 = lay.Activation(name=name)
        lay2 = lay.Activation.from_dump(lay1.dump)
        tensor = torch.rand(size=(7, 3, 8), dtype=torch.float)
        assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_AvgPool1d():
    lay1 = lay.AvgPool1d(6, stride=5)
    lay2 = lay.AvgPool1d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, 8, 32), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_AvgPool2d():
    lay1 = lay.AvgPool2d((6, 5), stride=(5, 4))
    lay2 = lay.AvgPool2d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, 8, 32, 37), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_BatchNorm1d():
    C = 6
    lay1 = lay.BatchNorm1d(C, affine=True, eps=0.01)
    lay1.weight = torch.nn.Parameter(torch.rand(size=(C,), dtype=torch.float))
    lay1.bias = torch.nn.Parameter(torch.rand(size=(C,), dtype=torch.float))
    lay1.running_mean = torch.rand(size=(C,), dtype=torch.float)
    lay1.running_var = torch.rand(size=(C,), dtype=torch.float)
    lay2 = lay.BatchNorm1d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, C, 8), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_BatchNorm2d():
    C = 6
    lay1 = lay.BatchNorm2d(C, affine=True, eps=0.01)
    lay1.weight = torch.nn.Parameter(torch.rand(size=(C,), dtype=torch.float))
    lay1.bias = torch.nn.Parameter(torch.rand(size=(C,), dtype=torch.float))
    lay1.running_mean = torch.rand(size=(C,), dtype=torch.float)
    lay1.running_var = torch.rand(size=(C,), dtype=torch.float)
    lay2 = lay.BatchNorm2d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, C, 8, 4), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_ConstantPad1d():
    C = 6
    lay1 = lay.ConstantPad1d((C, 3), value=-0.5)
    lay2 = lay.ConstantPad1d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, C, 8), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_ConstantPad2d():
    C = 6
    lay1 = lay.ConstantPad2d((C, 3), value=-0.5)
    lay2 = lay.ConstantPad2d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, C, 8, 10), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_Conv1d():
    C = 5
    lay1 = lay.Conv1d(C, 3, 7, stride=2)
    lay2 = lay.Conv1d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, C, 16), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_Conv2d():
    C = 5
    lay1 = lay.Conv2d(C, 3, (7, 3), stride=(3, 2))
    lay2 = lay.Conv2d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, C, 32, 16), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_PoolingStage1d():
    C = 5
    convolutions = [{"channels": 4, "window": 3},
                    {"channels": 5, "window": 6, "stacked": True},
                    {"channels": 6, "window": 2}]
    for padded in [True, False]:
        for pooling_type in [None, "Max", "Avg"]:
            lay1 = lay.PoolingStage1d(C, convolutions, padded=padded,
                                      pooling_type=pooling_type)
            lay2 = lay.PoolingStage1d.from_dump(lay1.dump)
            tensor = torch.rand(size=(7, C, 32), dtype=torch.float)
            assert torch.allclose(lay1(tensor), lay2(tensor),
                                  rtol=rtol, atol=atol)


def test_PoolingStage2d():
    C = 5
    convolutions = [{"channels": 4, "window": (3, 4)},
                    {"channels": 5, "window": (6, 3), "stacked": True},
                    {"channels": 6, "window": (2, 1)}]
    for padded in [True, False]:
        for pooling_type in [None, "Max", "Avg"]:
            lay1 = lay.PoolingStage2d(C, convolutions, padded=padded,
                                      pooling_type=pooling_type)
            lay2 = lay.PoolingStage2d.from_dump(lay1.dump)
            tensor = torch.rand(size=(7, C, 32, 16), dtype=torch.float)
            assert torch.allclose(lay1(tensor), lay2(tensor),
                                  rtol=rtol, atol=atol)


def test_Linear():
    C = 6
    lay1 = lay.Linear(C, 8)
    lay2 = lay.Linear.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, C), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_MaxPool1d():
    lay1 = lay.MaxPool1d(6, stride=5)
    lay2 = lay.MaxPool1d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, 8, 32), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_MaxPool2d():
    lay1 = lay.MaxPool2d((6, 5), stride=(5, 4))
    lay2 = lay.MaxPool2d.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, 8, 32, 37), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


def test_Encoder1d():
    C = 5
    convolutions = [{"channels": 4, "window": 3},
                    {"channels": 5, "window": 6, "stacked": True},
                    {"channels": 6, "window": 2}]
    pooling = [1, 3, 2]
    for padded in [True, False]:
        for stacked in [True, False]:
            lay1 = lay.Encoder1d(C, convolutions,
                                 pooling_windows=pooling,
                                 padded=padded,
                                 stacked=stacked)
            lay2 = lay.Encoder1d.from_dump(lay1.dump)
            tensor = torch.rand(size=(7, C, 16), dtype=torch.float)
            assert torch.allclose(lay1(tensor), lay2(tensor),
                                  rtol=rtol, atol=atol)


def test_Encoder2d():
    C = 5
    convolutions = [{"channels": 4, "window": (3, 4)},
                    {"channels": 5, "window": (6, 3), "stacked": True},
                    {"channels": 6, "window": (2, 1)}]
    pooling = [(2, 1), (1, 3), (2, 2)]
    for padded in [True, False]:
        for stacked in [True, False]:
            lay1 = lay.Encoder2d(C, convolutions,
                                 pooling_windows=pooling,
                                 padded=padded,
                                 stacked=stacked)
            lay2 = lay.Encoder2d.from_dump(lay1.dump)
            tensor = torch.rand(size=(7, C, 32, 16), dtype=torch.float)
            assert torch.allclose(lay1(tensor), lay2(tensor),
                                  rtol=rtol, atol=atol)


def test_UNet1d():
    C = 5
    convolutions = [{"channels": 4, "window": 3},
                    {"channels": 5, "window": 6, "stacked": True},
                    {"channels": 6, "window": 2}]
    pooling = [1, 3, 2]
    for padded in [True, False]:
        lay1 = lay.Encoder1d(C, convolutions,
                             pooling_windows=pooling,
                             padded=padded)
        lay2 = lay.Encoder1d.from_dump(lay1.dump)
        tensor = torch.rand(size=(7, C, 16), dtype=torch.float)
        assert torch.allclose(lay1(tensor), lay2(tensor),
                              rtol=rtol, atol=atol)


def test_UNet2d():
    C = 5
    convolutions = [{"channels": 4, "window": (3, 4)},
                    {"channels": 5, "window": (6, 3), "stacked": True},
                    {"channels": 6, "window": (2, 1)}]
    pooling = [(2, 1), (1, 3), (2, 2)]
    for padded in [True, False]:
        lay1 = lay.Encoder2d(C, convolutions,
                             pooling_windows=pooling,
                             padded=padded)
        lay2 = lay.Encoder2d.from_dump(lay1.dump)
        tensor = torch.rand(size=(7, C, 32, 16), dtype=torch.float)
        assert torch.allclose(lay1(tensor), lay2(tensor),
                              rtol=rtol, atol=atol)


def test_FullyConnected():
    C = 6
    lay1 = lay.FullyConnected(C, hidden_layers=[10, 5, 7])
    lay2 = lay.FullyConnected.from_dump(lay1.dump)
    tensor = torch.rand(size=(7, C), dtype=torch.float)
    assert torch.allclose(lay1(tensor), lay2(tensor), rtol=rtol, atol=atol)


if __name__ == "__main__":
    module = sys.modules[__name__]
    for attr in dir(module):
        if not attr.startswith("test_"):
            continue
        func = getattr(module, attr)
        print(attr)
        func()
