from typing import List


class Tokenizer:
    """
    A text tokenizer is an object with an 'encode' and a 'decode' method
    """

    def encode(self, sentence: str) -> List[int]:
        raise NotImplementedError()

    def decode(self, sentence: List[int]) -> str:
        raise NotImplementedError()


class DynamicTokenizer(Tokenizer):
    """
    A Dynamic tokenizer is a tokenizer that performs subword regularization
    at training time (sentence is segmented differently each iteration)
    """

    def encode(self, sentence: str, use_dropout: bool = False) -> List[int]:
        raise NotImplementedError()
