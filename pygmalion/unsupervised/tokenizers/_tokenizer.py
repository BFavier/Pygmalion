import torch
from typing import List, Iterable
from pygmalion.neural_networks._conversions import sentences_to_tensor


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

    def split(self, sentence: str) -> List[str]:
        """Returns the sentence splited token by token"""
        vocab = self.vocabulary
        return [vocab[i] for i in self.encode(sentence)]

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

    def encode(self, sentence: str, regularize: bool = False) -> List[int]:
        raise NotImplementedError()


class SpecialToken:
    """Special tokens for the <START>, <END>, and <PAD> tokens"""
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


class DynamicTextDataset:
    """
    Class emulating a torch.Tensor dataset
    for support to dynamic subword regularization
    """

    def __init__(self, text: Iterable[str], tokenizer: DynamicTokenizer,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.tokenizer = tokenizer
        self.text = list(text)

    def __getitem__(self, index):
        """allow indexing of the dataset"""
        if isinstance(index, int):
            return DynamicTextDataset([self.text[index]],
                                      self.tokenizer, device=self.device)
        elif isinstance(index, slice):
            return self.__getitem__(range(index.start or 0, index.stop,
                                          index.step or 1))
        else:
            return DynamicTextDataset([self.text[int(i)] for i in index],
                                      self.tokenizer, device=self.device)

    def __len__(self):
        """returns the length of the dataset"""
        return len(self.text)

    def to(self, device: torch.device):
        """return the DynamicTextDataset as stored on another device"""
        return DynamicTextDataset(self.text, self.tokenizer, device=device)

    def as_tensor(self, regularize: bool, max_length: int) -> torch.Tensor:
        """Returns the text dataset as a tensor"""
        return sentences_to_tensor(self.text, self.tokenizer, self.device,
                                   max_length=max_length,
                                   regularize=regularize)
