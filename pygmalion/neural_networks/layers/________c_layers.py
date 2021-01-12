from ..._library_loader import _lib
from ...tensor import Tensor as _Tensor
import ctypes as _c
import json as _json

# Typing of the Tensor functions
_lib.new_ConvolutionLayer.restype = _c.c_void_p
_lib.new_ConvolutionLayer.argtypes = [_c.c_char_p]

_lib.apply_ConvolutionLayer.restype = _c.c_void_p
_lib.apply_ConvolutionLayer.argtypes = [_c.c_void_p, _c.c_void_p]

_lib.del_ConvolutionLayer.restype = None
_lib.del_ConvolutionLayer.argtypes = [_c.c_void_p]

_lib.new_UpsamplingLayer.restype = _c.c_void_p
_lib.new_UpsamplingLayer.argtypes = [_c.c_char_p]

_lib.apply_UpsamplingLayer.restype = _c.c_void_p
_lib.apply_UpsamplingLayer.argtypes = [_c.c_void_p, _c.c_void_p, _c.c_void_p]

_lib.del_UpsamplingLayer.restype = None
_lib.del_UpsamplingLayer.argtypes = [_c.c_void_p]

_lib.new_DenseLayer.restype = _c.c_void_p
_lib.new_DenseLayer.argtypes = [_c.c_char_p]

_lib.apply_DenseLayer.restype = _c.c_void_p
_lib.apply_DenseLayer.argtypes = [_c.c_void_p, _c.c_void_p]

_lib.del_DenseLayer.restype = None
_lib.del_DenseLayer.argtypes = [_c.c_void_p]


# Defining the classes
class ConvolutionLayer:

    def __init__(self, data):
        if isinstance(data, _c.c_void_p):
            self.pointer = data
        elif isinstance(data, dict):
            settings = _json.dumps(data)
            self.pointer = _lib.new_ConvolutionLayer(settings.encode("utf-8"))
        else:
            raise TypeError(f"Unexpected type '{type(data)}'"
                            "in ConvolutionLayer's constructor")

    def __call__(self, input):
        assert isinstance(input, _Tensor)
        p = _lib.apply_ConvolutionLayer(self.pointer, input.pointer)
        return _Tensor(_c.c_void_p(p))

    def __del__(self):
        _lib.del_ConvolutionLayer(self.pointer)


class UpsamplingLayer:

    def __init__(self, data):
        if isinstance(data, _c.c_void_p):
            self.pointer = data
        elif isinstance(data, dict):
            settings = _json.dumps(data)
            self.pointer = _lib.new_UpsamplingLayer(settings.encode("utf-8"))
        else:
            raise TypeError(f"Unexpected type '{type(data)}'"
                            "in UpsamplingLayer's constructor")

    def __call__(self, input, concat):
        assert isinstance(input, _Tensor)
        p = _lib.apply_UpsamplingLayer(self.pointer, input.pointer,
                                       concat.pointer)
        return _Tensor(_c.c_void_p(p))

    def __del__(self):
        _lib.del_UpsamplingLayer(self.pointer)


class DenseLayer:

    def __init__(self, data):
        if isinstance(data, _c.c_void_p):
            self.pointer = data
        elif isinstance(data, dict):
            settings = _json.dumps(data)
            self.pointer = _lib.new_DenseLayer(settings.encode("utf-8"))
        else:
            raise TypeError(f"Unexpected type '{type(data)}'"
                            "in DenseLayer's constructor")

    def __call__(self, input):
        assert isinstance(input, _Tensor)
        p = _lib.apply_DenseLayer(self.pointer, input.pointer)
        return _Tensor(_c.c_void_p(p))

    def __del__(self):
        _lib.del_DenseLayer(self.pointer)
