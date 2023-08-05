import numpy as np
import pandas as pd
import torch
from typing import Any, Tuple, Iterable, Sequence


def split(*data: Tuple[Sequence], weights: Tuple[float] = (0.8, 0.2),
          shuffle: bool = True) -> tuple:
    """
    Splits the input data

    Parameters
    ----------
    data : tuple
        Tuple of iterables
    weights : tuple of float
        The fraction of testing data (internaly normalized to sum up to 1)
    shuffle : bool
        If True, the data is shuffled before splitting

    Returns
    -------
    tuple :
        tuples of data
    """
    L = len(data[0])
    indexes = np.random.permutation(L) if shuffle else np.arange(L)
    if any(f <= 0. for f in weights):
        raise ValueError("The weights must be superior to 0.")
    total = sum(weights)
    frac = tuple(w/total for w in weights)
    bounds = [int(round(sum(frac[:i])*L)) for i in range(len(frac)+1)]
    if len(data) > 1:
        splits = [tuple(_index(d, indexes[lower:upper]) for d in data)
                  for lower, upper in zip(bounds[:-1], bounds[1:])]
    else:
        splits = [_index(data[0], indexes[lower:upper])
                  for lower, upper in zip(bounds[:-1], bounds[1:])]
    return splits


def kfold(*data: Tuple[Iterable], k: int = 3, shuffle: bool = True) -> tuple:
    """
    Splits the input data into k-folds of (train, test) data

    Parameters
    ----------
    data : tuple
        Tuple of iterables
    k : int
        The number of folds to yield
    shuffle : bool
        If True, the data is shuffled before splitting


    Yields
    ------
    tuple :
        the (train, test) tuple of data
    """
    L = len(data[0])
    indexes = np.random.permutation(L) if shuffle else np.arange(L)
    indexes = np.array_split(indexes, k)
    for i in range(k):
        train_index = np.concatenate([ind for j, ind in enumerate(indexes)
                                      if j != i])
        train = tuple(_index(d, train_index) for d in data)
        test_index = indexes[i]
        test = tuple(_index(d, test_index) for d in data)
        yield train, test


def _index(data: Any, at: np.ndarray):
    """Indexes an input data. Method depends on it's type"""
    if data is None:
        return None
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.iloc[at]
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        return data[at]
    elif isinstance(data, Iterable):
        return [data[i] for i in at]
    else:
        raise RuntimeError(f"data type '{type(data)}' not supported")
