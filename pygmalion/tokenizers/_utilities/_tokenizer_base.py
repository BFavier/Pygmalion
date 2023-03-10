from typing import Optional, List, Tuple, Union, Iterable
from unidecode import unidecode
from pygmalion._model_base import ModelBase
from ._special_token import SpecialToken


class Tokenizer(ModelBase):
    """
    Tokenizer base is a base class for tokenizers

    Atributes
    ---------
    _ascii : bool
        whether or not to convert input to acii before tokenizing
    _lowercase : bool
        whether or not to convert input to lowercase before tokenizing
    _special_token_names : iterable of str
        name of the special tokens available with this tokenizer
    _vocabulary : tuple
        tuple of of all of the vocabulary, special tokens excluded
    _token_indexes : dict
        for a given token, gives it's index in the vocabulary
    """

    def __init__(self, ascii: bool, lowercase: bool, special_token_names: Iterable[str]):
        super().__init__()
        self._ascii = ascii
        self._lowercase = lowercase
        self._special_token_names = special_token_names
        self._vocabulary = tuple()
        self._token_indexes = dict()

    def __repr__(self) -> str:
        n_tokens = f"{len(self.vocabulary):,}".replace(",", " ")
        return f"{type(self).__name__}({n_tokens} tokens, ascii={self.ascii}, lowercase={self.lowercase}, special={self.special_tokens})"

    def __getattr__(self, attr):
        """
        indexes of special tokens in the vocabulary can be accessed as attributes
        """
        if attr in object.__getattribute__(self, "_special_token_names"):
            return object.__getattribute__(self, "_token_indexes")[SpecialToken(attr)]
        else:
            return object.__getattribute__(self, attr)

    def encode(self, string: str, start_token: bool = False,
               end_token: bool = False, padded_size: Optional[int] = None) -> List[int]:
        """
        Apply the tokenization
        """
        raise NotImplementedError()

    def decode(self, encoded: List[int]) -> str:
        """
        Decode a tokenized string
        """
        raise NotImplementedError()

    def split(self, string: str) -> List[bytes]:
        """
        Returns the string splited token by token
        """
        raise NotImplementedError()
    
    def _preprocess(self, string: str) -> str:
        """
        Apply ASCII conversion and lowercase conversion on strings if requested
        """
        if self.ascii:
            string = unidecode(string)
        if self.lowercase:
            string = string.lower()
        return string

    @property
    def vocabulary(self) -> Tuple[Union[str, bytes, SpecialToken], ...]:
        return self._vocabulary + self.special_tokens

    @property
    def special_tokens(self) -> Tuple[SpecialToken, ...]:
        return tuple(SpecialToken(name) for name in self._special_token_names)

    @special_tokens.setter
    def special_tokens(self, other: Iterable[Union[str, SpecialToken]]):
        self._special_token_names = tuple(token if isinstance(token, str) else token.name for token in other)

    @property
    def ascii(self) -> bool:
        return self._ascii

    @property
    def lowercase(self) -> int:
        return self._lowercase

    @property
    def n_tokens(self) -> int:
        return len(self._vocabulary) + len(self._special_token_names)