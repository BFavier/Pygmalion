from typing import List
from unidecode import unidecode
from ._utilities import Tokenizer

class AsciiCharTokenizer(Tokenizer):
    """
    Dummy Tokenizer that split text into single ascii characters tokens
    This tokenizer does not require training
    """

    @classmethod
    def from_dump(cls, dump: dict) -> "AsciiCharTokenizer":
        assert dump["type"] == cls.__name__
        return AsciiCharTokenizer()

    def __init__(self, lowercase: bool=False, special_tokens: List[str]=["START", "END", "PAD"]):
        super().__init__(ascii=True, lowercase=lowercase, special_token_names=special_tokens)
        self._vocabulary = [bytes([b]) for b in range(256)]

    def encode(self, string: str) -> List[int]:
        """
        encode a string
        """
        return list(self._preprocess(unidecode(string)).encode("utf-8"))

    def decode(self, encoded: List[int]) -> str:
        """
        decode an encoded string
        """
        return bytes(e for e in encoded if 0<= e < 256).decode("utf-8", errors="ignore")

    def split(self, string: str) -> List[bytes]:
        """
        split a string in bytes, with each byte beeing a token
        """
        return [b for b in self._preprocess(unidecode(string))]

    @property
    def vocabulary(self):
        return tuple(bytes([i]) for i in range(256))

    @property
    def dump(self):
        return {"type": type(self).__name__}
