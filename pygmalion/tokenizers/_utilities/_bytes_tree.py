import random
from itertools import chain
from typing import Iterable, Optional, Tuple, List, Union


class BytesTree:
    """
    BytesTree is a representation of all know bytes tokens
    """

    def __init__(self, vocabulary: Iterable[bytes]=[bytes([i]) for i in range(256)]):
        self.data = {}
        for v in vocabulary:
            self.push(v)
    
    def __str__(self) -> str:
        return "\n".join(str(b) for b in self.vocabulary)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
    
    def __iter__(self):
        return iter(self.vocabulary)

    def split(self, item: bytes, p_dropout: Optional[float]=None) -> List[bytes]:
        """
        split a document into the longest known tokens
        """
        assert isinstance(item, bytes)
        return list(self._split(item, p_dropout))

    def push(self, item: bytes):
        """
        learn a new token
        """
        assert isinstance(item, bytes)
        prefix, suffix, leaf = self._propagate(self.data, item, None)
        if suffix != b"":
            leaf[suffix] = {}

    @property
    def vocabulary(self) -> Tuple[bytes]:
        """
        returns the list of known tokens
        """
        return tuple(self._get_vocabulary(self.data, b""))

    def _propagate(self, data: dict, value: bytes, p_dropout: Optional[float]) -> Tuple[bytes, bytes, dict]:
        """
        recursively returns a (prefix, suffix, leaf) tuple
        with:
            'prefix' the bytes that matched until now
            'suffix' the bytes that did not match
            'leaf' the dictionary where the suffix should be appended
        """
        if (p_dropout is None) or (data is self.data) or (random.random() >= p_dropout):
            for k, v in data.items():
                if value.startswith(k):
                    prefix, suffix, leaf = self._propagate(v, value[len(k):], p_dropout)
                    return (k+prefix), suffix, leaf
        return b"", value, data

    def _split(self, item: bytes, p_dropout: Optional[None] = None) -> Iterable[bytes]:
        """
        split a bytes object by known sequence
        """
        while len(item) > 0:
            prefix, suffix, leaf = self._propagate(self.data, item, p_dropout)
            item = suffix
            yield prefix

    def _get_vocabulary(self, data: dict, prefix: bytes) -> Iterable[str]:
        """
        recursively returns all the tokens contained in the given data
        """
        if data is not self.data:
            yield prefix
        for k, v in data.items():
            for b in self._get_vocabulary(v, prefix+k):
                yield b
