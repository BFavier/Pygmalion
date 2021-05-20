import re
from random import random
from array import array
from itertools import chain, count
from collections import Counter, deque
from typing import List, Tuple, Iterable, Iterator, Dict


class Token:

    def __repr__(self) -> str:
        return f"#{self.i}"

    def __int__(self) -> int:
        return self.i

    def __init__(self, i):
        self.i = i
        self.previous = None
        self.next = None


class Pair:

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

    def __repr__(self):
        return "".join(repr(t) for t in self)

    def __init__(self, sentence: str):
        self.sentence = sentence
        data = (int(b) for b in sentence.encode("utf-8"))
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
            removed_pairs.update([tuple(merge_pair)])
            previous, next = merge_pair.previous, merge_pair.next
            merge_pair.unlink()
            if previous is not None:
                left = previous.first
                previous.unlink()
                removed_pairs.update([tuple(previous)])
                self._unregister_pair(previous)
                new_pair = Pair(left, token)
                added_pairs.update([tuple(new_pair)])
                self._register_pair(new_pair)
            else:
                self.first = token
            if next is not None:
                right = next.second
                next.unlink()
                removed_pairs.update([tuple(next)])
                self._unregister_pair(next)
                new_pair = Pair(token, right)
                added_pairs.update([tuple(new_pair)])
                self._register_pair(new_pair)


class Corpus:

    def __init__(self):
        self.code = dict()

    def train(self, sentences: List[str]):
        sentences = [Sentence(s) for s in sentences]
        pairs_count = self._get_pair_counts(sentences)
        print("training", flush=True)
        for i in count(1):
            if i > 100:
                break
            best_pair, pair_count = max(pairs_count.items(),
                                        key=lambda x: x[1])
            new_token = 256 + len(self.code)
            self.code[new_token] = best_pair
            added_pairs, removed_pairs = Counter(), Counter()
            for s in sentences:
                s.merge(best_pair, new_token, added_pairs, removed_pairs)
            pairs_count += added_pairs
            pairs_count -= removed_pairs
        return sentences

    def encode(self, sentence: str, dropout: float = 0.) -> List[int]:
        sentence = list(sentence.encode("utf-8"))
        for t, c in self.code.items():
            sentence = list(self._contract(sentence, c, t))
        return sentence

    def decode(self, encoded: List[int]) -> str:
        codes = [coding for coding in self.code.items()]
        for t, c in codes[::-1]:
            encoded = self._expand(encoded, t, c)
        return bytes(encoded).decode("utf-8", errors="replace")

    def _prune_code(self):
        pass

    def _get_pair_counts(self, sentences: List[Sentence]):
        iterable = (Counter({k: len(p) for k, p in s.pairs.items()})
                    for s in sentences)
        return sum(iterable, Counter())

    def _expand(self, sentence: Iterable[int], token: int,
                code: Tuple[int]) -> Iterator[int]:
        for t in sentence:
            if t == token:
                for c in code:
                    yield c
            else:
                yield(t)

    def _contract(self, sentence: List[int], code: Tuple[int],
                  token: int) -> Iterator[int]:
        i = 0
        while i < len(sentence):
            j = i+len(code)
            if j <= len(sentence) and tuple(sentence[i:j]) == code:  # TODO: don't create a tuple for nothing, use all(...)
                i = j
                yield token
            else:
                yield sentence[i]
                i += 1


if __name__ == "__main__":
    from timeit import timeit
    import pathlib
    import IPython
    path = pathlib.Path(__file__).parent
    with open(path / "corpus.txt", "r") as file:
        corpus = file.read().lower().split("\n")
    # with open(path / "europarl-v7.en.txt", "r", encoding="latin-1") as file:
        # corpus = file.read().lower().split("\n")
    # bpe = BytePairEncoder()
    # res = bpe.train(corpus)
    # corp = Corpus(corpus)
    # print(bpe.vocabulary)
    # coded = [bpe.encode(c) for c in corpus]
    # s = Sentence("1223333")
    # s.merge((50, 50), 0)
    # s.merge((49, 0), 1)
    c = Corpus()
    res = c.train(corpus)
    s = c.encode("Hello world")
    c.decode(s)
    # print(timeit(c.train(corpus, as_array=True), number=10))
    # print(timeit(c.train(corpus, as_array=False), number=10))
    IPython.embed()
