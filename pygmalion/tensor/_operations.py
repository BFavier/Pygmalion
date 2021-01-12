from .._library_loader import _lib
from ._tensor import Tensor as _Tensor
import ctypes as _c

# Typing of the layers functions
_lib.pad.restype = _c.c_void_p
_lib.pad.argtypes = [_c.c_void_p, _c.c_double,
                     _c.c_uint, _c.c_uint,
                     _c.c_uint, _c.c_uint]

_lib.convolve.restype = _c.c_void_p
_lib.convolve.argtypes = [_c.c_void_p, _c.c_void_p, _c.c_void_p,
                          _c.c_uint, _c.c_uint]

_lib.max_pool.restype = _c.c_void_p
_lib.max_pool.argtypes = [_c.c_void_p,
                          _c.c_uint, _c.c_uint]

_lib.batch_normalize.restype = _c.c_void_p
_lib.batch_normalize.argtypes = [_c.c_void_p, _c.c_void_p,
                                 _c.c_void_p, _c.c_void_p, _c.c_void_p]

_lib.linear.restype = _c.c_void_p
_lib.linear.argtypes = [_c.c_void_p, _c.c_void_p, _c.c_void_p]

_lib.resample.restype = _c.c_void_p
_lib.resample.argtypes = [_c.c_void_p, _c.c_uint, _c.c_uint]

_lib.Tensor_tanh.restype = _c.c_void_p
_lib.Tensor_tanh.argtypes = [_c.c_void_p]

_lib.Tensor_relu.restype = _c.c_void_p
_lib.Tensor_relu.argtypes = [_c.c_void_p]


# defining the functions
def pad(input: _Tensor, value: float, size: tuple):
    assert isinstance(input, _Tensor) and (input.ndims == 3)
    left, right, top, bottom = size
    return _Tensor(_c.c_void_p(_lib.pad(input.pointer, value, left,
                                        right, top, bottom)))


def convolve(input: _Tensor, kernel: _Tensor, bias: _Tensor, stride: tuple):
    assert isinstance(input, _Tensor) and (input.ndims == 3)
    assert isinstance(kernel, _Tensor) and (kernel.ndims == 4)
    assert isinstance(bias, _Tensor) and (bias.ndims == 1)
    si, sk = input.shape[-2:], kernel.shape[-2:]
    assert all([k <= i for i, k in zip(si, sk)])
    S_h, S_w = stride
    return _Tensor(_c.c_void_p(_lib.convolve(input.pointer,
                                             kernel.pointer,
                                             bias.pointer,
                                             S_h, S_w)))


def max_pool(input: _Tensor, window: tuple):
    assert isinstance(input, _Tensor) and (input.ndims == 3)
    height, width = window
    return _Tensor(_c.c_void_p(_lib.max_pool(input.pointer, height, width)))


def batch_normalize(input: _Tensor, mean: _Tensor, variance: _Tensor,
                    weight: _Tensor, bias: _Tensor):
    assert isinstance(input, _Tensor) and (input.ndims == 3)
    assert isinstance(mean, _Tensor) and (mean.ndims == 1)
    assert isinstance(variance, _Tensor) and (variance.ndims == 1)
    assert isinstance(weight, _Tensor) and (weight.ndims == 1)
    assert isinstance(bias, _Tensor) and (bias.ndims == 1)
    return _Tensor(_c.c_void_p(_lib.batch_normalize(input.pointer,
                                                    mean.pointer,
                                                    variance.pointer,
                                                    weight.pointer,
                                                    bias.pointer)))


def linear(input: _Tensor, weight: _Tensor, bias: _Tensor):
    assert isinstance(input, _Tensor) and (input.ndims == 1)
    assert isinstance(weight, _Tensor) and (weight.ndims == 2)
    assert isinstance(bias, _Tensor) and (bias.ndims == 1)
    return _Tensor(_c.c_void_p(_lib.linear(input.pointer,
                                           weight.pointer,
                                           bias.pointer)))


def resample(input: _Tensor, new_height: int, new_width: int) -> _Tensor:
    assert (new_height > 0) and (new_width > 0)
    return _Tensor(_c.c_void_p(_lib.resample(input.pointer,
                                             new_height,
                                             new_width)))


def tanh(input: _Tensor):
    assert isinstance(input, _Tensor)
    return _Tensor(_c.c_void_p(_lib.Tensor_tanh(input.pointer)))


def relu(input: _Tensor):
    assert isinstance(input, _Tensor)
    return _Tensor(_c.c_void_p(_lib.Tensor_relu(input.pointer)))


def identity(input: _Tensor) -> _Tensor:
    return input


# class Linear:
#     def __init__(self, parameters: dict):
#         self.weight = _Tensor(parameters["weights"])
#         if parameters["bias"] is None:
#             self.bias = _Tensor([0.]*self.weights.shape[0])
#         else:
#             self.bias = _Tensor(parameters["bias"])

#     def __call__(self, input) -> _Tensor:
#         return linear(input, self.weight, self.bias)


# class Padding:
#     def __init__(self, parameters: dict):
#         self.value = parameters["value"]
#         self.size = parameters["size"]

#     def __call__(self, input: _Tensor) -> _Tensor:
#         return pad(input, self.value, self.size)


# class Convolution:
#     def __init__(self, parameters: dict):
#         self.weights = _Tensor(parameters["weights"])
#         if parameters["bias"] is None:
#             self.bias = _Tensor([0]*self.weights.shape[0])
#         else:
#             self.bias = _Tensor(parameters["bias"])
#         self.stride = parameters["stride"]

#     def __call__(self, input: _Tensor) -> _Tensor:
#         return convolve(input, self.weights, self.bias, self.stride)


# class BatchNorm:
#     def __init__(self, parameters: dict):
#         self.mean = _Tensor(parameters["running_mean"])
#         self.variance = _Tensor(parameters["running_var"])
#         self.weight = _Tensor(parameters["weight"])
#         self.bias = _Tensor(parameters["bias"])

#     def __call__(self, input: _Tensor) -> _Tensor:
#         return batch_normalize(input, self.mean, self.variance,
#                                self.weight, self.bias)


# class MaxPool:
#     def __init__(self, parameters: dict):
#         self.window = parameters["window"]

#     def __call__(self, input: _Tensor) -> _Tensor:
#         return max_pool(input, self.window)


# class NonLinear:
#     def __init__(self, name: str):
#         if name is None:
#             name = "identity"
#         module = _sys.modules[__name__]
#         if not hasattr(module, name):
#             raise ValueError(f"non-linear function '{name}' "
#                              "is not implemented")
#         self.func = getattr(module, name)

#     def __call__(self, input: _Tensor) -> _Tensor:
#         return self.func(input)


# class ConvolutionLayer(_m.Model):
#     def __init__(self, dump):
#         super().__init__(dump)
#         self.padding = Padding(self.parameters["padding"])
#         self.convolution = Convolution(self.parameters["convolution"])
#         self.batch_norm = BatchNorm(self.parameters["batch_norm"])
#         self.pooling = MaxPool(self.parameters["pooling"])
#         self.non_linear = NonLinear(self.parameters["non_linear"])

#     def __call__(self, input: _Tensor) -> _Tensor:
#         T = self.padding(input)
#         T = self.convolution(T)
#         T = self.batch_norm(T)
#         T = self.pooling(T)
#         return self.non_linear(T)


# class DenseLayer(_m.Model):
#     def __init__(self, dump):
#         super().__init__(dump)
#         self.linear = Linear(self.parameters["linear"])
#         self.non_linear = NonLinear(self.parameters["non_linear"])

#     def __call__(self, input: _Tensor) -> _Tensor:
#         T = self.linear(input)
#         return self.non_linear(T)
