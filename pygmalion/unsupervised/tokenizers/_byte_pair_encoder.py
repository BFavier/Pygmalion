import re
from random import random
from itertools import count, chain
from collections import Counter, deque
from typing import List, Tuple, Iterable, Iterator, Dict, Optional
from ._tokenizer import DynamicTokenizer


class Token:
    """
    A token is a word from a vocabulary.
    It is linked into a sentence by the previous and next pair of tokens.

    Attributes
    ----------
    previous : Pair or None
        the pair leading to the previous token in the sentence
    next : Pair or None
        the pair leading to the next token in the sentence
    """

    def __repr__(self) -> str:
        return f"#{self.i}"

    def __int__(self) -> int:
        return self.i

    def __init__(self, i):
        self.i = i
        self.previous = None
        self.next = None


class Pair:
    """
    Pair is a pair of tokens

    Attributes
    ----------
    first : Token
        the first token
    second : Token
        the second token
    """

    def __init__(self, first: Token, second: Token):
        first.next = self
        second.previous = self
        self.first = first
        self.second = second

    def __iter__(self) -> tuple:
        yield int(self.first)
        yield int(self.second)

    def __eq__(self, other: 'Token') -> bool:
        return self is other

    def __ne__(self, other: 'Token') -> bool:
        return not self == other

    def unlink(self):
        self.first.next = None
        self.second.previous = None

    @property
    def previous(self) -> 'Pair':
        return self.first.previous

    @property
    def next(self) -> 'Pair':
        return self.second.next


class Sentence:
    """
    A sentence is a series of tokens linked by tokens pairs

    Attributes
    ----------
    pairs : dict of {(int, int): [Token, ...]}
        the pairs present in the sentence
    first : Token
        the first token of the sentence (a sentence can't be empty)
    weight : int
        the number of occurences of the sentence in the corpus
        (to compute merges only once)
    """

    def __repr__(self):
        return "".join(repr(t) for t in self)

    def __init__(self, sentence: str, weight: int = 1):
        """
        sentence : str
            the sentence to encode
        weight : int
            the sentence weight (if a sentence is repeated in the corpus)
        """
        assert len(sentence) > 0
        data = (int(b) for b in sentence.encode("utf-8"))
        self.weight = weight
        self.pairs = dict()
        self.first = Token(next(data))
        previous = self.first
        for b in data:
            new = Token(b)
            self._register_pair(Pair(previous, new))
            previous = new

    def __iter__(self):
        token = self.first
        yield token
        link = token.next
        while link is not None:
            token = link.second
            yield token
            link = token.next

    def _register_pair(self, pair: Pair):
        key = tuple(pair)
        pairs = self.pairs.get(key, None)
        if pairs is None:
            self.pairs[key] = deque([pair])
        else:
            pairs.append(pair)

    def _unregister_pair(self, pair):
        key = tuple(pair)
        self.pairs[key].remove(pair)

    def merge(self, pair: Tuple[int, int], new_token: int,
              added_pairs: Counter, removed_pairs: Counter):
        """
        Replace all occurences of the 'pair' in the sentence by 'new_token'.
        Also count each pair that is added in the process,
        and each pair that is removed.
        """
        pairs = self.pairs.get(pair, [])
        while len(pairs) > 0:
            token = Token(new_token)
            merge_pair = pairs.popleft()
            removed_pairs[tuple(merge_pair)] += self.weight
            previous, next = merge_pair.previous, merge_pair.next
            merge_pair.unlink()
            if previous is not None:
                left = previous.first
                previous.unlink()
                removed_pairs[tuple(previous)] += self.weight
                self._unregister_pair(previous)
                new_pair = Pair(left, token)
                added_pairs[tuple(new_pair)] += self.weight
                self._register_pair(new_pair)
            else:
                self.first = token
            if next is not None:
                right = next.second
                next.unlink()
                removed_pairs[tuple(next)] += self.weight
                self._unregister_pair(next)
                new_pair = Pair(token, right)
                added_pairs[tuple(new_pair)] += self.weight
                self._register_pair(new_pair)


class BytePairEncoder(DynamicTokenizer):
    """
    byte level Byte Pair Encoding (BPE) is a method of subword tokenization

    Attributes
    ----------
    code : dict
        the dictionary of {token: (subtokens, ...)}
    dropout : float or None
        the probability to skip a byte pair merge at encoding time,
        only in training mode. This improves model's robustness to typos
    """

    @classmethod
    def from_dump(cls, dump: dict) -> "BytePairEncoder":
        assert dump["type"] == cls.__name__
        return cls(code=dump["code"], dropout=dump["dropout"])

    def __repr__(self):
        return (f"{type(self).__name__}({len(self.vocabulary)} tokens,"
                f" dropout={self.dropout})")

    def __init__(self, code: Dict[int, Tuple[int, ...]] = dict(),
                 dropout: Optional[float] = None):
        """
        Build a BytePairEncoder tokenizer

        Parameters
        ----------
        code : dict of {int: tuple of int}
            a dict of {token: subtokens}
        dropout : float or None
            either None (no dropout used) or a float between 0 and 1
            the dropout is the probability of a byte pair merge to be skipped
        """
        self.code = code
        self.dropout = dropout

    def train(self, sentences: List[str], max_tokens: int = 5000,
              min_frequency: float = 1.0E-4, verbose: bool = True,
              word_piece: bool = True):
        """
        Trains the byte pair encoding

        Parameters
        ----------
        sentences : list of str
            the list of sentences
        max_tokens : int
            the maximum number of tokens in the resulting vocabulary
        min_frequency : float
            the minimum frequency of each new token in the corpus to be valid
        verbose : bool
            If True, display progression
        word_piece : bool
            If True, the corpus is split into
            single words/white spaces/punctuation,
            and subword can't cross the word boundary.
            This should be set to False for languages that are not whitespace
            separated.

        Returns
        -------
        dict of {bytes: int}:
            the count of each subword occurence in the training corpus
        """
        print("loading data ...", flush=True)
        if word_piece:
            unique_words = Counter(self._split_words(sentences))
            sentences = [Sentence(s, weight=c) for s, c in unique_words.items()
                         if len(s) > 0]
        else:
            sentences = [Sentence(s) for s in sentences if len(s) > 0]
        code = dict()
        pairs_count = self._get_pair_counts(sentences)
        tokens_count = self._get_tokens_count(sentences)
        n_tokens = sum(tokens_count.values())
        n_tokens = self._build_code(code, sentences, pairs_count, tokens_count,
                                    n_tokens, max_tokens, min_frequency,
                                    verbose)
        self.code, final_tokens = self._prune_code(code, tokens_count,
                                                   n_tokens, min_frequency,
                                                   verbose)
        return {self._bytes(k, self.code): i for k, i in final_tokens.items()}

    def encode(self, sentence: str, use_dropout: bool = True) -> List[int]:
        """
        Apply the tokenization
        """
        sentence = list(sentence.encode("utf-8"))
        for t, c in self.code.items():
            sentence = list(self._contract(sentence, c, t,
                            dropout=self.dropout if use_dropout else None))
        return sentence

    def decode(self, encoded: List[int]) -> str:
        """
        Decode a tokenized sentence
        """
        codes = [coding for coding in self.code.items()]
        for t, c in codes[::-1]:
            encoded = self._expand(encoded, t, c)
        return bytes(encoded).decode("utf-8", errors="replace")

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "code": self.code,
                "dropout": self.dropout}

    @property
    def vocabulary(self) -> List[bytes]:
        """returns all the single tokens"""
        byte = [bytes([i]) for i in range(256)]
        return byte + [self._bytes(t, self.code) for t in self.code.keys()]

    @property
    def n_tokens(self):
        return len(self.code) + 256

    def _split_words(self, sentences: Iterable[str]) -> Iterable[str]:
        """Split each sentence into a list of 'words'"""
        return chain(*(re.findall(r"[\w]+|[^\w]", s) for s in sentences))

    def _bytes(self, token_index: int, code: Dict[int, Tuple[int]]) -> bytes:
        """returns the bytes representation of a token"""
        if token_index < 256:
            return bytes([token_index])
        else:
            return b"".join((self._bytes(t, code)
                             for t in code[token_index]))

    def _build_code(self, code: Dict[int, Tuple[int]],
                    sentences: List[Sentence],
                    pairs_count: Dict[Tuple[int, int], int],
                    tokens_count: Dict[int, int], n_tokens: int,
                    max_tokens: int, min_frequency: float, verbose: bool
                    ) -> int:
        """
        Fills the code dictionnary with new token until not possible anymore
        """
        for i in count(1):
            if len(pairs_count) == 0:
                if verbose:
                    print("\nno more pairs to merge", flush=True)
                break
            best_pair, pair_count = max(pairs_count.items(),
                                        key=lambda x: x[1])
            new_token = 256 + len(code)
            new_token_frequency = pair_count / (n_tokens - pair_count)
            if new_token_frequency < min_frequency:
                if verbose:
                    print("\nminimum token frequency reached", flush=True)
                break
            code[new_token] = best_pair
            added_pairs, removed_pairs = Counter(), Counter()
            for s in sentences:
                s.merge(best_pair, new_token, added_pairs, removed_pairs)
            pairs_count += added_pairs
            pairs_count -= removed_pairs
            tokens_count[best_pair[0]] -= pair_count
            tokens_count[best_pair[1]] -= pair_count
            tokens_count[new_token] += pair_count
            n_tokens -= pair_count
            n_valid = sum(self._token_is_valid(token, tokens_count,
                                               n_tokens, min_frequency)
                          for token in code.keys()) + 256
            if verbose:
                print(f"\r\033[K\rMerge iterations {i}: "
                      f"{n_valid} tokens, "
                      f"{new_token_frequency:.3g} token frequency",
                      end="", flush=True)
            if n_valid >= max_tokens:
                if verbose:
                    print("\nmaximum number of tokens reached", flush=True)
                break
        return n_tokens

    def _prune_code(self, code, tokens_count: Dict[int, int], n_tokens: int,
                    min_frequency: float, verbose: bool) -> tuple:
        """
        Remove tokens that are too unfrequent from the code dictionary
        Returns the pruned code, and pruned tokens count
        """
        for i in count(1):
            for token in tuple(code.keys()):
                if not self._token_is_valid(token, tokens_count,
                                            n_tokens, min_frequency):
                    for t in code[token]:
                        tokens_count[t] += tokens_count[token]
                    n_tokens += tokens_count[token]*(len(code[token]) - 1)
                    tokens_count.pop(token)
                    code = self._unmerge_tokens(token, code)
                    break
            else:
                break
            if verbose:
                print(f"\r\033[K\rPrunning iteration {i}",
                      end="", flush=True)
        mapping = {k: i+256 for i, k in enumerate(code.keys())}
        mapping.update({i: i for i in range(256)})
        final_code = {mapping[k]: tuple(mapping[t] for t in v)
                      for k, v in code.items()}
        final_tokens = {mapping[k]: i for k, i in tokens_count.items()
                        if i > 0}
        if verbose:
            print("")
        return final_code, final_tokens

    def _get_pair_counts(self, sentences: List[Sentence]) -> Counter:
        """
        Returns a counter of the occurences of each pair in all the sentences
        """
        iterable = (Counter({k: len(p)*s.weight for k, p in s.pairs.items()})
                    for s in sentences)
        return sum(iterable, Counter())

    def _get_tokens_count(self, sentences: List[Sentence]) -> Counter:
        """
        Returns a counter of the occurences of each token in all the sentences
        """
        iterable = (Counter({int(t): s.weight for t in s}) for s in sentences)
        return sum(iterable, Counter())

    def _unmerge_tokens(self, token: int, code: Dict[int, Tuple[int]]
                        ) -> Dict[int, Tuple[int]]:
        """
        return a 'code' without 'token'
        and with all it's occurences replaced by it's own code
        """
        token_code = code[token]
        return {t: tuple(self._expand(v, token, token_code))
                for t, v in code.items() if t != token}

    def _token_is_valid(self, token: int, tokens_count: Dict[int, int],
                        n_tokens: int, min_frequency: float) -> bool:
        """
        Return True if the token is valid (frequent enough)
        """
        return tokens_count[token]/n_tokens > min_frequency

    def _expand(self, sentence: Iterable[int], token: int,
                code: Tuple[int]) -> Iterator[int]:
        """
        substitue the 'token' by the 'code' in the 'sentence'

        Example
        -------
        >>> list(self._expand((1, 2, 3, 4, 3), 3, (1, 10, 100)))
        [1, 2, 1, 10, 100, 4, 1, 10, 100]
        """
        for t in sentence:
            if t == token:
                for c in code:
                    yield c
            else:
                yield(t)

    def _contract(self, sentence: List[int], code: Tuple[int],
                  token: int, dropout: float = 0.) -> Iterator[int]:
        """
        replace occurences of the given 'code' by the 'value'
        in a 'sentence'

        Example
        -------
        >>> list(self._contract([1, 2, 3, 4, 3], (2, 3, 4), 1000))
        [1, 1000, 3]
        """
        i = 0
        while i < len(sentence):
            j = i+len(code)
            if (j <= len(sentence) and tuple(sentence[i:j]) == code
                    and (dropout is None or random() >= dropout)):
                i = j
                yield token
            else:
                yield sentence[i]
                i += 1
