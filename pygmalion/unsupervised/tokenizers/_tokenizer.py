from typing import List


class Tokenizer:
    """
    A text tokenizer is an object with an 'encode' and a 'decode' method
    """

    def encode(self, sentence: str) -> List[int]:
        """encode a sentence"""
        raise NotImplementedError()

    def decode(self, sentence: List[int]) -> str:
        """decode an encoded sentence"""
        raise NotImplementedError()

    @property
    def vocabulary(self):
        """Returns all the unique tokens known by the tokenizer"""
        raise NotImplementedError()

    @property
    def n_tokens(self):
        """number of tokens known by the tokenizer"""
        raise NotImplementedError()


class DynamicTokenizer(Tokenizer):
    """
    A Dynamic tokenizer is a tokenizer that performs subword regularization
    at training time (sentence is segmented differently each iteration)
    """

    def encode(self, sentence: str, use_dropout: bool = False) -> List[int]:
        raise NotImplementedError()
