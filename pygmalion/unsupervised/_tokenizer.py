import re
from itertools import chain
from collections import Counter
from typing import Iterable, List, Dict


class WhitespaceTokenizer:
    """
    Tokenizer for whitespace separated words

    Attributes
    ----------
    vocabulary : list of str
        set of unique possible words, including '#UNKNOWN#' for unknow words
    """

    _unknown = "#UNKNOWN#"

    @classmethod
    def from_dump(cls, dump: dict) -> "WhitespaceTokenizer":
        assert dump["type"] == cls.__name__
        return WhitespaceTokenizer(vocabulary=dump["vocabulary"])

    def __repr__(self):
        return f"{type(self).__name__}({len(self.vocabulary)} words)"

    def __init__(self, vocabulary: List[str] = []):
        """build a tokenizer from the given vocabulary"""
        if self._unknown not in vocabulary:
            vocabulary += [self._unknown]
        self.vocabulary = vocabulary

    def train(self, corpus: Iterable[str], max_tokens: int = 20000,
              min_frequency: float = 1.0E-6) -> Dict[str, float]:
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
        self.vocabulary = vocab + [self._unknown]
        frequencies = {w: words_count[w]/n_words for w in vocab}
        frequencies[self._unknown] = 1 - sum(frequencies.values())
        return frequencies

    def encode(self, sentence: str) -> List[int]:
        """encode a sentence"""
        words = {w: i for i, w in enumerate(self.vocabulary)}
        return [words[self._tokenized(w)] for w in self._split_words(sentence)]

    def decode(self, sentence: List[int]) -> str:
        """decode a sentence"""
        return " ".join([self.vocabulary[i] for i in sentence])

    def _tokenized(self, word: str) -> str:
        """replace the word by the 'unknown' token if unknown"""
        return word if word in self.vocabulary else self._unknown

    def _split_words(self, sentence: str) -> List[str]:
        """Split each sentence into a list of 'words'"""
        return re.findall(r"[\w]+|[^\s\w]", sentence)

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "vocabulary": self.vocabulary}
