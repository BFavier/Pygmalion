# import torch
# from pygmalion.neural_networks._conversions import sentences_to_tensor
from pygmalion._model import Model
from typing import List


class Tokenizer(Model):
    """
    A text tokenizer is an object with an 'encode' and a 'decode' method
    """

    def encode(self, sentence: str, regularize: bool = False) -> List[int]:
        """encode a sentence"""
        raise NotImplementedError()

    def decode(self, sentence: List[int]) -> str:
        """decode an encoded sentence"""
        raise NotImplementedError()

    def split(self, sentence: str, regularize: bool = False) -> List[str]:
        """Returns the sentence splited token by token"""
        vocab = self.vocabulary
        return [vocab[i] for i in self.encode(sentence, regularize)]

    @property
    def vocabulary(self):
        """Returns all the unique tokens known by the tokenizer"""
        raise NotImplementedError()

    @property
    def n_tokens(self):
        """number of tokens known by the tokenizer"""
        raise NotImplementedError()

    @property
    def jit(self):
        """
        Returns True if the tokenizer performs subword regularization
        and requires 'Just In Time' tokenization
        (tokenization will be different at each epoch)
        """
        return False


class SpecialToken:
    """
    Special tokens for the <START>, <END>, <PAD>, <UNKNOWN>... tokens
    """
    def __repr__(self):
        return f"<{self.name}>"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        is_token = issubclass(type(other), type(self))
        return is_token and (self.name == other.name)

    def __init__(self, name: str):
        self.name = name
