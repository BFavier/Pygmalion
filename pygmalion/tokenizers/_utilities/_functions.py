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


_split_words_pattern = re.compile(r"(\d+|[^\W\d]+|([^\s\w])\2*)")


def split_words(string: str) -> List[str]:
    """
    Extract all sequences of letters/digits/same punctuation while ignoring spaces

    Example
    -------
    >>> split_words("horizon@hotmail.fr 14h30(+20m) ...")
    ['horizon', '@', 'hotmail', '.', 'fr', '14', 'h', '30', '(', '+', '20', 'm', ')', '...']
    """
    return [m[0] for m in _split_words_pattern.findall(string)]


_split_wordpiece_pattern = re.compile(r"\S+\s?")


def split_wordpiece(string: str) -> List[str]:
    """
    Extract all series of digits or series of letters from each string

    Example
    -------
    >>> split_wordpiece("horizon@hotmail.fr 14h30(+20m) ...")
    ['horizon@hotmail.fr ', '14h30(+20m) ', '...']
    """
    return _split_wordpiece_pattern.findall(string)


if __name__ == "__main__":
    import IPython
    IPython.embed()
