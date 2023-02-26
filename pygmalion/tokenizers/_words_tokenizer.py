import re
from itertools import chain
from collections import Counter
from typing import Iterable, List, Dict
from ._utilities import split_words, TokenizerBase, SpecialToken


class WordsTokenizer(TokenizerBase):
    """
    Tokenizer for whitespace separated words, with additional

    Attributes
    ----------
    vocabulary : list of str
        set of unique possible words, including '#UNKNOWN#' for unknow words
    """

    @classmethod
    def from_dump(cls, dump: dict) -> "WordsTokenizer":
        assert dump["type"] == cls.__name__
        vocabulary = tuple(dump["vocabulary"]) + (cls._unknown,)
        return WordsTokenizer(vocabulary=vocabulary)

    def __init__(self, vocabulary: List[str] = [],
                 ascii: bool=False, lowercase: bool=False,
                 special_tokens: List[str]=["UNKNOWN", "START", "END", "PAD"]):
        super().__init__(ascii, lowercase, special_tokens)
        self.vocabulary = vocabulary

    def fit(self, corpus: Iterable[str], max_tokens: int = 20000,
            min_frequency: float = 1.0E-6) -> Dict[str, int]:
        """
        find all unique words from a corpus of whitespace separated sentences
        """
        words = (split_words(s) for s in corpus)
        words_count = Counter(chain(*words))
        n_words = sum(words_count.values())
        vocab = sorted((w for w, c in words_count.items()
                        if c/n_words > min_frequency),
                       key=lambda w: words_count[w], reverse=True)
        vocab = vocab[:max_tokens]
        vocab_count = {k: words_count[k] for k in vocab}
        vocab_count = dict(sorted(vocab_count.items(),
                                  key=lambda item: item[1],
                                  reverse=True))
        self.vocabulary = vocab
        n_unknowns = n_words - sum(vocab_count.values())
        vocab_count = dict(chain([(self.UNKNOWN, n_unknowns)],
                                 vocab_count.items()))
        return vocab_count

    def encode(self, string: str) -> List[int]:
        """encode a string"""
        return [self._token_indexes.get(w, self.UNKNOWN)
                for w in split_words(self._preprocess(string))]

    def decode(self, encoded: List[int]) -> str:
        """decode an encoded string"""
        vocab = self.vocabulary
        return " ".join([str(vocab[i]) for i in encoded])

    def split(self, string: str) -> List[str]:
        """split a string"""
        return [w if w in self._token_indexes.keys() else SpecialToken("UNKNOWN")
                for w in split_words(self._preprocess(string))]

    @TokenizerBase.vocabulary.setter
    def vocabulary(self, other):
        self._vocabulary = tuple(t for t in other if isinstance(t, str))
        self._token_indexes = {w: i for i, w in enumerate(self.vocabulary)}

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "vocabulary": self._vocabulary}
