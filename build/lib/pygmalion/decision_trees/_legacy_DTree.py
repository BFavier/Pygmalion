import ctypes as _c
import numpy as _np
from .._library_loader import _lib

# defining structures
c_double_p = _c.POINTER(_c.c_double)


class cBranch(_c.Structure):
    _fields_ = [("is_leaf", _c.c_bool),
                ("child_true", _c.c_uint),
                ("child_false", _c.c_uint),
                ("gain", _c.c_double),
                ("split_numerical_x", _c.c_bool),
                ("i", _c.c_uint),
                ("j", _c.c_uint),
                ("y_num", _c.c_double),
                ("iy_cat", _c.c_uint)]


class cTree(_c.Structure):
    _fields_ = [("n_branches", _c.c_uint),
                ("branches", _c.POINTER(cBranch))]


# delete tree
_lib.delete_tree.restype = None
_lib.delete_tree.argtype = [cTree]


def delete_tree(ctree: cTree):
    _lib.delete_tree(ctree)


# grow regressor_tree
_lib.grow_regressor.restype = cTree
_lib.grow_regressor.argtype = [c_double_p, _c.c_char_p, c_double_p, _c.c_uint,
                               _c.c_uint, _c.c_uint, _c.c_uint, _c.c_uint,
                               _c.c_uint, _c.c_uint, c_double_p]


def grow_regressor_tree(num_x: _np.ndarray, cat_x: _np.ndarray, y: _np.ndarray,
                        y_pred: _np.ndarray = None,
                        min_samples: int = 1, max_depth: int = 1000,
                        max_leafs: int = 1000):
    assert isinstance(num_x, _np.ndarray)
    assert isinstance(cat_x, _np.ndarray)
    assert isinstance(y, _np.ndarray)
    assert cat_x.dtype.char == "U" or cat_x.dtype.char == "S"
    n_rows = len(y)
    _, n_numerical_columns = num_x.shape
    _, n_categorical_columns = cat_x.shape
    n_chars_x = cat_x.dtype.itemsize
    num_x = num_x.ctypes.data_as(c_double_p)
    cat_x = cat_x.ctypes.data_as(_c.c_char_p)
    y = y.ctypes.data_as(c_double_p)
    if isinstance(y_pred, _np.ndarray):
        y_pred = y_pred.ctypes.data_as(c_double_p)
    else:
        y_pred = None
    return _lib.grow_regressor(num_x, cat_x, y, n_rows, n_numerical_columns,
                               n_categorical_columns, n_chars_x, min_samples,
                               max_depth, max_leafs, y_pred)


# grow classifier_tree
_lib.grow_regressor.restype = cTree
_lib.grow_regressor.argtype = [c_double_p, _c.c_char_p, _c.c_char_p, _c.c_uint,
                               _c.c_uint, _c.c_uint, _c.c_uint, _c.c_uint,
                               _c.c_uint, _c.c_uint, _c.c_uint, _c.c_char_p]


def grow_classifier_tree(num_x: _np.ndarray, cat_x: _np.ndarray,
                         y: _np.ndarray,
                         min_samples: int = 1, max_depth: int = 1000,
                         max_leafs: int = 1000):
    assert isinstance(num_x, _np.ndarray)
    assert isinstance(cat_x, _np.ndarray)
    assert isinstance(y, _np.ndarray)
    assert cat_x.dtype.char == "U" or cat_x.dtype.char == "S"
    assert y.dtype.char == "U" or y.dtype.char == "S"
    n_rows = len(y)
    _, n_numerical_columns = num_x.shape
    _, n_categorical_columns = cat_x.shape
    n_chars_x = cat_x.dtype.itemsize
    n_chars_y = y.dtype.itemsize
    num_x = num_x.ctypes.data_as(c_double_p)
    cat_x = cat_x.ctypes.data_as(_c.c_char_p)
    y = y.ctypes.data_as(_c.c_char_p)
    return _lib.grow_classifier(num_x, cat_x, y, n_rows, n_numerical_columns,
                                n_categorical_columns, n_chars_x, n_chars_y,
                                min_samples, max_depth, max_leafs, None)
