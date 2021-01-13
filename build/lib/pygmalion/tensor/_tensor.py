from .._library_loader import _lib
import ctypes as _c
import numpy as _np

# Typing of the Tensor functions
_lib.new_Tensor.restype = _c.c_void_p
_lib.new_Tensor.argtypes = [_c.POINTER(_c.c_double),
                            _c.POINTER(_c.c_uint),
                            _c.c_uint]

_lib.Tensor_copy.restype = _c.c_void_p
_lib.Tensor_copy.argtypes = [_c.c_void_p]

_lib.Tensor_reshape.restype = _c.c_void_p
_lib.Tensor_reshape.argtypes = [_c.c_void_p,
                                _c.POINTER(_c.c_uint),
                                _c.c_uint]

_lib.Tensor_ndims.restype = _c.c_uint
_lib.Tensor_ndims.argtypes = [_c.c_void_p]

_lib.Tensor_shape.restype = _c.POINTER(_c.c_uint)
_lib.Tensor_shape.argtypes = [_c.c_void_p]

_lib.Tensor_data.restype = _c.POINTER(_c.c_double)
_lib.Tensor_data.argtypes = [_c.c_void_p]

_lib.Tensor_set.restype = None
_lib.Tensor_set.argtypes = [_c.c_void_p, _c.POINTER(_c.c_uint), _c.c_double]

_lib.Tensor_get.restype = _c.c_double
_lib.Tensor_get.argtypes = [_c.c_void_p, _c.POINTER(_c.c_uint)]

_lib.Tensor_subTensor.restype = _c.c_void_p
_lib.Tensor_subTensor.argtypes = [_c.c_void_p, _c.c_uint]

_lib.del_Tensor.restype = None
_lib.del_Tensor.argtypes = [_c.c_void_p]

_lib.Tensor_concatenate.restype = _c.c_void_p
_lib.Tensor_concatenate.argtypes = [_c.POINTER(_c.c_void_p),
                                    _c.c_uint, _c.c_uint]


# Defining the class
class Tensor:

    def concatenate(tensors: list, axis: int = 0) -> 'Tensor':
        assert axis >= 0
        assert len(tensors) > 0
        arr = (_c.c_void_p * len(tensors))(*[t.pointer for t in tensors])
        return Tensor(_c.c_void_p(_lib.Tensor_concatenate(arr,
                                                          len(tensors), axis)))

    def __init__(self, data):
        if isinstance(data, _c.c_void_p):
            self.pointer = data
        elif hasattr(data, "__iter__") and not isinstance(data, str):
            arr = _np.array(data, dtype=_np.double, order="C")
            shape = arr.shape
            ndims = len(shape)
            shape = (_c.c_uint * ndims)(*shape)
            p = arr.ctypes.data_as(_c.POINTER(_c.c_double))
            self.pointer = _lib.new_Tensor(p, shape, ndims)
        else:
            # Set the poinetr to nullptr so that the finalizer '__del__'
            # doesn't raise an error
            self.pointer = None
            # Then raise an error
            raise TypeError(f"Unexpected type '{type(data)}'"
                            "in Tensor's constructor")

    def __del__(self):
        _lib.del_Tensor(self.pointer)

    def __getitem__(self, key):
        S = self.shape
        if (type(key) == int) and (len(S) == 1):
            key = (key,)
        T = type(key)
        if T == int:
            if (key >= S[0]) or (key < 0):
                raise IndexError("Indexing tensor out of range")
            p = _c.c_void_p(_lib.Tensor_subTensor(self.pointer, key))
            return Tensor(p)
        elif T == tuple:
            if len(key) != len(S):
                raise IndexError("Index size and tensor shape dont match")
            index = (_c.c_uint*len(S))(*key)
            return _lib.Tensor_get(self.pointer, index)
        else:
            raise TypeError(f"Unexpected index type '{T}'")

    def __setitem__(self, key, value):
        S = self.shape
        if not hasattr(key, "__iter__"):
            key = (key,)
        if len(key) != len(S):
            raise IndexError("Index size and tensor shape don't match")
        assert [isinstance(k, int) for k in key]
        assert [k >= 0 and k < s for k, s in zip(key, S)]
        index = (_c.c_uint * len(key))(*key)
        _lib.Tensor_set(self.pointer, index, value)

    def __repr__(self):
        return f"Tensor(address={self.pointer}, shape={self.shape})"

    def copy(self):
        """Returns a deep copy of the tensor"""
        return Tensor(_c.c_void_p(_lib.Tensor_copy(self.pointer)))

    def reshape(self, new_shape: list):
        """Returns the same tensor reshaped (underlying data is shared)"""
        assert _np.product(new_shape) == _np.product(self.shape)
        ndims = len(new_shape)
        new_shape = (_c.c_uint * ndims)(*new_shape)
        return Tensor(_c.c_void_p(_lib.Tensor_reshape(self.pointer,
                                                      new_shape, ndims)))

    def flatten(self):
        """Returns the tensor reshaped as a 1D tensor"""
        return self.reshape([_np.product(self.shape)])

    @property
    def numpy(self):
        """Returns the tensor as a numpy array (memory is not shared)"""
        S = self.shape
        address = _lib.Tensor_data(self.pointer)
        c_array = _c.cast(address, _c.POINTER(_c.c_double*_np.product(S)))[0]
        return _np.array(c_array, order="C").reshape(S)

    @property
    def shape(self):
        """size of each dimension of the tensor"""
        _shape = _lib.Tensor_shape(self.pointer)
        return [_shape[i] for i in range(self.ndims)]

    @property
    def ndims(self):
        """number of dimensions of the tensor"""
        return _lib.Tensor_ndims(self.pointer)
