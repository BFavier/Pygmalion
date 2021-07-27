from itertools import chain
from collections import Counter
from typing import Iterable, List, Dict
from ._tokenizer import Tokenizer, SpecialToken


class CharTokenizer(Tokenizer):
    """
    Tokenizer that split text into single characters
    """

    _unknown = SpecialToken("UNKNOWN")

    @classmethod
    def from_dump(cls, dump: dict) -> "CharTokenizer":
        assert dump["type"] == cls.__name__
        vocabulary = [cls._unknown]+dump["vocabulary"]
        return CharTokenizer(vocabulary=vocabulary)

    def __repr__(self):
        return f"{type(self).__name__}({len(self.vocabulary)} chars)"

    def __init__(self, vocabulary: List[str] = []):
        """build a tokenizer from the given vocabulary"""
        if self._unknown not in vocabulary:
            vocabulary += [self._unknown]
        self.vocabulary = vocabulary

    def train(self, corpus: Iterable[str]) -> Dict[str, int]:
        """
        find all unique words from a corpus of whitespace separated sentences
        """
        char_count = Counter(chain(*corpus))
        char_count = dict(sorted(char_count.items(),
                                 key=lambda item: item[1],
                                 reverse=True))
        self.vocabulary = [self._unknown] + list(char_count.keys())
        return char_count

    def encode(self, sentence: str, regularize: bool = False) -> List[int]:
        """encode a sentence"""
        return [self._char_indexes.get(c, 0) for c in sentence]

    def decode(self, sentence: List[int]) -> str:
        """decode a sentence"""
        return " ".join([str(self.vocabulary[i]) for i in sentence
                         if i < self.n_tokens])

    @property
    def vocabulary(self):
        return self._vocabulary

    @vocabulary.setter
    def vocabulary(self, other):
        self._vocabulary = other
        self._char_indexes = {w: i for i, w in enumerate(self.vocabulary)}

    @property
    def n_tokens(self):
        return len(self.vocabulary)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "vocabulary": self.vocabulary[1:]}
