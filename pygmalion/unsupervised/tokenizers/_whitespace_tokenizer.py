import re
from itertools import chain
from collections import Counter
from typing import Iterable, List, Dict
from ._tokenizer import Tokenizer, SpecialToken


class WhitespaceTokenizer(Tokenizer):
    """
    Tokenizer for whitespace separated words

    Attributes
    ----------
    vocabulary : list of str
        set of unique possible words, including '#UNKNOWN#' for unknow words
    """

    _unknown = SpecialToken("UNKNOWN")

    @classmethod
    def from_dump(cls, dump: dict) -> "WhitespaceTokenizer":
        assert dump["type"] == cls.__name__
        vocabulary = [cls._unknown]+dump["vocabulary"]
        return WhitespaceTokenizer(vocabulary=vocabulary)

    def __repr__(self):
        return f"{type(self).__name__}({len(self.vocabulary)} words)"

    def __init__(self, vocabulary: List[str] = []):
        """build a tokenizer from the given vocabulary"""
        if self._unknown not in vocabulary:
            vocabulary += [self._unknown]
        self.vocabulary = vocabulary

    def train(self, corpus: Iterable[str], max_tokens: int = 20000,
              min_frequency: float = 1.0E-6) -> Dict[str, int]:
        """
        find all unique words from a corpus of whitespace separated sentences
        """
        words = (self._split_words(s) for s in corpus)
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
        self.vocabulary = [self._unknown] + vocab
        n_unknowns = n_words - sum(vocab_count.values())
        vocab_count = dict(chain([(self._unknown, n_unknowns)],
                                 vocab_count.items()))
        return vocab_count

    def encode(self, sentence: str, regularize: bool = False) -> List[int]:
        """encode a sentence"""
        return [self._word_indexes.get(w, 0)
                for w in self._split_words(sentence)]

    def decode(self, sentence: List[int]) -> str:
        """decode a sentence"""
        return " ".join([str(self.vocabulary[i]) for i in sentence
                         if i < self.n_tokens])

    def _split_words(self, sentence: str) -> List[str]:
        """Split each sentence into a list of 'words'"""
        return re.findall(r"[\w]+|[^\s\w]", sentence)

    @property
    def vocabulary(self):
        return self._vocabulary

    @vocabulary.setter
    def vocabulary(self, other):
        self._vocabulary = other
        self._word_indexes = {w: i for i, w in enumerate(self.vocabulary)}

    @property
    def n_tokens(self):
        return len(self.vocabulary)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "vocabulary": self.vocabulary[1:]}
