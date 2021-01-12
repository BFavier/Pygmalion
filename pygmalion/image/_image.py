from .._library_loader import _lib
from ..tensor import Tensor as _Tensor
import ctypes as _c
import numpy as _np
import matplotlib.pyplot as _plt
import os as _os

# Typing of the Tensor functions
_lib.new_Image_from_path.restype = _c.c_void_p
_lib.new_Image_from_path.argtypes = [_c.c_char_p]

_lib.new_Image_from_data.restype = _c.c_void_p
_lib.new_Image_from_data.argtypes = [_c.c_uint8, _c.c_uint64, _c.c_uint64,
                                     _c.c_uint64]

_lib.Image_height.restype = _c.c_uint64
_lib.Image_height.argtypes = [_c.c_void_p]

_lib.Image_width.restype = _c.c_uint64
_lib.Image_width.argtypes = [_c.c_void_p]

_lib.Image_channels.restype = _c.c_uint64
_lib.Image_channels.argtypes = [_c.c_void_p]

_lib.Image_as_grayscale.restype = _c.c_void_p
_lib.Image_as_grayscale.argtypes = [_c.c_void_p]

_lib.Image_as_RGB.restype = _c.c_void_p
_lib.Image_as_RGB.argtypes = [_c.c_void_p]

_lib.Image_as_RGBA.restype = _c.c_void_p
_lib.Image_as_RGBA.argtypes = [_c.c_void_p]

_lib.Image_resized.restype = _c.c_void_p
_lib.Image_resized.argtypes = [_c.c_void_p, _c.c_uint64, _c.c_uint64]

_lib.Image_save.restype = None
_lib.Image_save.argtypes = [_c.c_void_p]

_lib.Image_copy.restype = None
_lib.Image_copy.argtypes = [_c.c_void_p]

_lib.Image_as_Tensor.restype = _c.c_void_p
_lib.Image_as_Tensor.argtypes = [_c.c_void_p]

_lib.del_Image.restype = None
_lib.del_Image.argtypes = [_c.c_void_p]


# Defining the class
class Image:

    def __init__(self, data):
        if isinstance(data, _c.c_void_p):
            self.pointer = data
        elif isinstance(data, str):
            assert _os.path.isfile(data)
            self.pointer = _lib.new_Image_from_path(data.encode("utf-8"))
        elif hasattr(data, "__iter__"):
            data = _np.array(data, dypes=_np.uint8)
            if len(data.shape) == 2:
                data = data[:, :, None]
            assert len(data.shape) == 3
            height, width, channels = data.shape
            input = data.ctypes.as_type(_c.c_uint8)
            self.pointer = _lib.new_Image_from_data(input, height, width,
                                                    channels)
        else:
            self.pointer = None
            raise TypeError(f"Unexpected type '{type(data)}'"
                            "in Image's constructor")

    def __del__(self):
        _lib.del_Image(self.pointer)

    def draw(self, axis=None):
        if axis is None:
            _, axis = _plt.subplots()
        axis.imshow(self.numpy, cmap="gray")
        axis.set_xticks([])
        axis.set_yticks([])

    def as_grayscale(self):
        return Image(_c.c_void_p(_lib.Image_as_grayscale(self.pointer)))

    def as_RGB(self):
        return Image(_c.c_void_p(_lib.Image_as_RGB(self.pointer)))

    def as_RGBA(self):
        return Image(_c.c_void_p(_lib.Image_as_RGBA(self.pointer)))

    def resized(self, height: int, width: int):
        return Image(_c.c_void_p(_lib.Image_resized(self.pointer,
                                                    height, width)))

    def save(self, path: str):
        _lib.Image_save(path.encode("utf-8"))

    @property
    def tensor(self):
        return _Tensor(_c.c_void_p(_lib.Image_as_Tensor(self.pointer)))

    @property
    def numpy(self):
        arr = self.tensor.numpy
        if arr.shape[0] == 1:
            return arr.reshape(arr.shape[1:]).astype(_np.uint8)
        else:
            return _np.moveaxis(self.tensor.numpy, 0, -1).astype(_np.uint8)
