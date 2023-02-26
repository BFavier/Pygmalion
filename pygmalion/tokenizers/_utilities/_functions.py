from typing import Iterable, Any, Tuple


def zip_pairs(iterable: Iterable[Any]) -> Iterable[Tuple[Any, Any]]:
    """
    returns an iterator over pairs

    Example
    -------
    >>> list(zip_pairs(range(6)))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    """
    first, second = iter(iterable), iter(iterable)
    next(second, None)
    return zip(first, second)