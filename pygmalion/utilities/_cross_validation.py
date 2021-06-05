import numpy as np
import pandas as pd
from typing import Any, Tuple, Iterable


def split(*data: Tuple[Any], frac: float = 0.2, shuffle: bool = True) -> tuple:
    """
    Splits the input data in two (train, test)

    Parameters
    ----------
    data : tuple
        Tuple of iterables
    frac : float
        The fraction of testing data
    shuffle : bool
        If True, the data is shuffled before splitting

    Returns
    -------
    tuple :
        the 'first' and 'second' tuples of data
    """
    L = len(data[0])
    indexes = np.random.permutation(L) if shuffle else np.arange(L)
    limit = int(round(frac * L))
    b = indexes[:limit]
    a = indexes[limit:]
    train = [_index(d, a) for d in data]
    test = [_index(d, b) for d in data]
    return tuple(train), tuple(test)


def kfold(*data: Tuple[Any], k: int = 3, shuffle: bool = True) -> tuple:
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
    elif isinstance(data, np.ndarray):
        return data[at]
    elif isinstance(data, Iterable):
        return [data[i] for i in at]
    else:
        raise RuntimeError(f"data type '{type(data)}' not supported")
