from typing import List


class CharTokenizer:
    """
    Tokenizer that split text into single bytes chunks
    (using UTF-8 encoding)
    """

    @classmethod
    def from_dump(cls, dump: dict) -> "CharTokenizer":
        assert dump["type"] == cls.__name__
        return CharTokenizer()

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __init__(self):
        pass

    def encode(self, sentence: str, regularize: bool = False) -> List[int]:
        """encode a sentence"""
        return list(sentence.encode("utf-8"))

    def decode(self, sentence: List[int]) -> str:
        """decode a sentence"""
        return bytes(sentence).decode("utf-8", errors="ignore")

    @property
    def vocabulary(self):
        return [bytes([i]) for i in range(256)]

    @property
    def n_tokens(self):
        return 256

    @property
    def dump(self):
        return {"type": type(self).__name__}
