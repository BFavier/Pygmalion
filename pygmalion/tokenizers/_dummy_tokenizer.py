from typing import List
from ._utilities import Tokenizer


class DummyTokenizer(Tokenizer):
    """
    Dummy Tokenizer that split text into bytes of it's utf-8 encoding.
    This tokenizer does not require training.
    """

    @classmethod
    def from_dump(cls, dump: dict) -> "DummyTokenizer":
        assert dump["type"] == cls.__name__
        return DummyTokenizer()

    def __init__(self, ascii: bool=False, lowercase: bool=False,
                 special_tokens: List[str]=["START", "END", "PAD"]):
        """
        Parameters
        ----------
        ascii : bool
            If True, the text is unidecoded as a first preprocessing step
        lowercase : bool
            If True, the text is put to lower case as a preprocessing step
        special tokens
        """
        super().__init__(ascii=ascii, lowercase=lowercase, special_tokens=special_tokens)

    def encode(self, string: str) -> List[int]:
        """
        encode a string
        """
        return list(string.encode("utf-8"))

    def decode(self, encoded: List[int]) -> str:
        """
        decode an encoded string
        """
        return bytes(encoded).decode("utf-8", errors="ignore")

    def split(self, string: str) -> List[bytes]:
        """
        split a string in bytes, with each byte beeing a token
        """
        return [b for b in string.encode("utf-8")]

    @property
    def vocabulary(self):
        return tuple(bytes([i]) for i in range(256)) + tuple(self.special_tokens)

    @property
    def n_tokens(self) -> int:
        return len(self.vocabulary)

    @property
    def dump(self):
        return {"type": type(self).__name__}
