from typing import Any, Tuple, List, Iterable, Optional, Dict, Union
from collections import Counter
from unidecode import unidecode
import random
import pathlib
import re
import json
from ._utilities import SpecialToken, zip_pairs, BytesTree


class BytePairEncoder:

    @classmethod
    def load(cls, file: str) -> 'BytePairEncoder':
        """
        Load a model from the disk (must be a .json)

        Parameters
        ----------
        file : str
            path of the file to read
        """
        file = pathlib.Path(file)
        if not file.is_file():
            raise FileNotFoundError(f"The file '{file}' does not exist")
        suffix = file.suffix.lower()
        if suffix == ".json":
            with open(file) as json_file:
                dump = json.load(json_file)
        else:
            raise ValueError("The file must be '.json' file, "
                             f"but got a '{suffix}'")
        return cls.from_dump(dump)

    @classmethod
    def from_dump(cls, dump: dict) -> "BytePairEncoder":
        assert dump["type"] == cls.__name__
        kwargs = dict(dump)
        kwargs.pop("type")
        code = {int(k): v for k, v in kwargs.pop("code").items()}
        return cls(code=code, **kwargs)

    def __getattr__(self, attr):
        if attr in object.__getattribute__(self, "_special_token_names"):
            return object.__getattribute__(self, "_word_indexes")[SpecialToken(attr)]
        else:
            return object.__getattribute__(self, attr)

    def __repr__(self):
        return (f"{type(self).__name__}({len(self._vocabulary)} tokens,"
                f" dropout={self.dropout})")

    def __init__(self, code: Dict[int, Tuple[int, ...]] = {i: [i] for i in range(256)},
                 dropout: Optional[float] = None, ascii: bool = False,
                 lowercase: bool = False, special_tokens: Iterable[str] = ["PAD"]):
        """
        Build a BytePairEncoder tokenizer

        Parameters
        ----------
        code : dict of {int: tuple of int}
            a dict of {token: subtokens}
        dropout : float or None
            either None (no dropout used) or a float between 0 and 1
            the dropout is the probability of a byte pair merge to be skipped
            during encoding
        ascii : bool
            If True, the text is converted to ascii before tokenizing.
            Warning: if True, the decoded encoded result is not necesserly
            equal to the input, and the number of bytes might not be preserved.
        lowercase : bool
            If True, the text is converted to lowercase before tokenizing
        special_tokens : iterable of str
            all the special tokens available with the tokenizer
        """
        self.dropout = dropout
        self._ascii = ascii
        self._lowercase = lowercase
        self.special_tokens = special_tokens
        self.code = dict(code)

    def fit(self, batch_generator: Iterable[List[str]], max_vocabulary_size: int = 5000,
            min_frequency: float = 1.0E-6, verbose: bool = True, 
            pre_tokenize: bool = False, count_duplicates: bool = False):
        """
        Trains the byte pair encoding

        Parameters
        ----------
        batch_generator : list of str
            A generator that yields batches of list of strings.
            Each string is a string to tokenize
            Each list yielded is a sample batch to compute byte pair frequencies on
        max_vocabulary_size : int
            the maximum number of tokens in the resulting vocabulary
        min_frequency : float
            the minimum frequency of each new token in the corpus to be valid
        verbose : bool
            If True, display progression
        pre_tokenize : bool
            If True, each string is splited into
            single words/white spaces/punctuation,
            and subword can't cross the word boundary.
            This should be set to False for languages that are not whitespace
            separated.
        count_duplicates : bool
            Usefull if tokenizing at word level in a "word piece" fashion.
            Count occurence of each unique string in the batch to speed up the
            algorithm if some strings are repeated many times in a batch.
        """
        try:
            for i, batch in enumerate(batch_generator):
                if len(self.code) >= max_vocabulary_size:
                    if verbose:
                        print("\nmaximum number of tokens reached", flush=True)
                    break
                if pre_tokenize:
                    batch = self._pre_tokenize(batch)
                if count_duplicates:
                    sequences_count = Counter(batch)
                    sequences = [self.split(unique, with_dropout=True) for unique in sequences_count.keys()]
                    weights = sequences_count.values()
                else:
                    sequences = [self.split(string, with_dropout=True) for string in batch]
                    weights = None
                n_tokens = sum(len(seq) * w for seq, w in zip(sequences, weights or [1]*len(sequences)))
                pairs = self._pairs_count(sequences, weights)
                if len(pairs) == 0:
                    if verbose:
                        print("\nno more pairs to merge", flush=True)
                    break
                best_pair, pair_count = max(pairs.items(), key=lambda x: x[1])
                new_token = len(self.code)
                new_token_frequency = pair_count / (n_tokens - pair_count)
                if new_token_frequency < min_frequency:
                    if verbose:
                        print("\nminimum token frequency reached", flush=True)
                    break
                self.code[new_token] = [self._word_indexes[b] for b in best_pair]
                new_token_bytes = b"".join(best_pair)
                self._vocabulary.append(new_token_bytes)
                self._word_indexes[new_token_bytes] = len(self._vocabulary) - 1
                self._bytes_tree.push(new_token_bytes)
                if verbose:
                    print(f"\r\033[K\rMerge iteration {i}: "
                          f"{len(self.code)} tokens, "
                          f"new token frequency={new_token_frequency:.3g}",
                          end="", flush=True)
        except KeyboardInterrupt:
            print("\nInterupted by the user")
        self.code = self.code  # update all hidden attributes in case the user cut some


    def encode(self, string: str, with_dropout: bool = True,
               start_token: bool = False, end_token: bool = False,
               padded_size: Optional[int] = None) -> List[int]:
        """
        Apply the tokenization
        """
        string = [self._word_indexes[token] for token in self.split(string, with_dropout)]
        if start_token:
            string.insert(0, self.START)
        if end_token:
            string.append(self.END)
        if padded_size is not None:
            if len(string) > padded_size:
                raise ValueError(f"Cannot pad string of size {len(string)}"
                                 f" to size {padded_size}")
            string.extend([self.PAD]*(padded_size-len(string)))
        return string

    def decode(self, encoded: List[int]) -> str:
        """
        Decode a tokenized string
        """
        vocabulary = self.vocabulary
        subwords = [vocabulary[i] for i in encoded]
        decoded = b"".join(b for b in subwords if isinstance(b, bytes))
        return decoded.decode("utf-8", errors="replace")

    def split(self, string: str, with_dropout: bool = True) -> List[bytes]:
        """Returns the string splited token by token"""
        if self.ascii:
            string = unidecode(string)
        if self.lowercase:
            string = string.lower()
        return self._bytes_tree.split(string.encode("utf-8"), p_dropout=self.dropout if with_dropout else None)

    def save(self, file: str, overwrite: bool = True):
        """
        Saves a model to the disk (as .json)

        Parameters
        ----------
        file : str
            The path where the file must be created
        overwritte : bool
            If True, the file is overwritten
        """
        file = pathlib.Path(file)
        path = file.parent
        suffix = file.suffix.lower()
        if not path.is_dir():
            raise ValueError(f"The directory '{path}' does not exist")
        if not(overwrite) and file.exists():
            raise FileExistsError(f"The file '{file}' already exists,"
                                  " set 'overwrite=True' to overwrite.")
        if suffix == ".json":
            with open(file, "w") as json_file:
                json.dump(self.dump, json_file)
        else:
            raise ValueError("The model must be saved as a '.json' "
                             f" file, but got '{suffix}'")

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "code": self.code,
                "dropout": self.dropout,
                "ascii": self.ascii,
                "lowercase": self.lowercase,
                "special_tokens": self._special_token_names}

    @property
    def code(self) -> Dict[int: List[int]]:
        return self._code

    @code.setter
    def code(self, other):
        self._code = other
        # setting vocabulary
        code_bytes = {i: bytes([i]) for i in range(256)}
        not_represented = {i: c for i, c in self.code.items() if i not in code_bytes.keys()}
        while len(not_represented) > 0:
            tmp = {}
            for i, c in not_represented.items():
                if all(j in code_bytes.keys() for j in c):
                    code_bytes[i] = b"".join(code_bytes[j] for j in c)
                else:
                    tmp[i] = c
            not_represented = tmp

        self._vocabulary = tuple(code_bytes.values()) + self.special_tokens
        # setting word indexes
        self._word_indexes = {w: i for i, w in enumerate(self.vocabulary)}
        # setting the BytesTree
        self._bytes_tree = BytesTree(sorted(self._vocabulary, key=lambda x: len(x)))

    @property
    def vocabulary(self) -> Tuple[Union[bytes, SpecialToken], ...]:
        return self._vocabulary

    @property
    def special_tokens(self) -> Tuple[SpecialToken, ...]:
        return tuple(SpecialToken(name) for name in self._special_token_names)
    
    @special_tokens.setter
    def special_tokens(self, other: Iterable[Union[str, SpecialToken]]):
        self._special_token_names = tuple(token if isinstance(token, str) else token.name for token in other)
        self._vocabulary = tuple(bytes(k) for k in self.code.keys()) + self.special_tokens

    @property
    def n_tokens(self):
        return len(self.code) + len(self._special_token_names)

    @property
    def ascii(self) -> bool:
        return self._ascii

    @property
    def lowercase(self) -> int:
        return self._lowercase
    
    @staticmethod
    def _pre_tokenize(batch: Iterable[str]) -> Iterable[str]:
        """
        Extract all series of digits or series of letters from each string

        Example
        -------
        >>> list(self._pre_tokenize(["Tökenizer2000, stârts_at 14h30..."]))
        ['Tökenizer', '2000', ',', 'stârts', 'at', '14', 'h', '30', '...']
        """
        return (token for string in batch for token in
                re.findall(r"[\d]+ ?|[^\W\d]+ ?|[^\w\s]+ ?", string))

    def _bytes(self, token_index: int, code: Dict[int, Tuple[int]]) -> bytes:
        """
        returns the bytes representation of a token from a (potentially unordered) code
        """
        if token_index < 256:
            return bytes([token_index])
        else:
            return b"".join((self._bytes(t, code) for t in code[token_index]))

    @staticmethod
    def _pairs_count(sequences: List[List[bytes]],
                     weights: Optional[List[float]]) -> Counter:
        """
        returns a Counter of all pairs encountered in the tokens sequences
        """
        if weights is None:
            return Counter(pair for sequence in sequences for pair in zip_pairs(sequence))
        else:
            counter = Counter()
            for weight, sequence in zip(weights, sequences):
                for pair in zip_pairs(sequence):
                    counter[pair] += weight
            return counter
