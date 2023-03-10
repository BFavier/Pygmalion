from typing import List
from ._utilities import Tokenizer

class CharTokenizer(Tokenizer):
    """
    Tokenizer that split text into single bytes tokens (using UTF-8 encoding)
    This dummy tokenizer does not compress information
    but also does not require training
    """

    @classmethod
    def from_dump(cls, dump: dict) -> "CharTokenizer":
        assert dump["type"] == cls.__name__
        return CharTokenizer()

    def __init__(self, ascii: bool, lowercase: bool,
                 special_tokens: List[str]=["START", "END", "PAD"]):
        super().__init__(ascii, lowercase, special_tokens)

    def encode(self, string: str) -> List[int]:
        """
        encode a string
        """
        return list(self._preprocess(string).encode("utf-8"))

    def decode(self, encoded: List[int]) -> str:
        """
        decode an encoded string
        """
        return bytes(encoded).decode("utf-8", errors="ignore")

    def split(self, string: str) -> List[bytes]:
        """
        split a string in bytes, with each byte beeing a token
        """
        return [b for b in self._preprocess(string).encode("utf-8")]

    @property
    def vocabulary(self):
        return tuple(bytes([i]) for i in range(256))

    @property
    def dump(self):
        return {"type": type(self).__name__}
