import re
from typing import Iterable, Any, Tuple, List


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


_wordpiece_pattern = re.compile(r"[\d]+ ?|[^\W\d]+ ?|[^\w\s]+ ?")


def split_wordpiece(string: str) -> List[str]:
    """
    Extract all series of digits or series of letters from each string

    Example
    -------
    >>> split_string("stârts_at 14h30 ...")
    ['stârts_at ', '14', 'h', '30 ', '...']
    """
    return _wordpiece_pattern.findall(string)
