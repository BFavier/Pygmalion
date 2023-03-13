from typing import List, Tuple
import torch
import numpy as np


class RomanNumeralsGenerator:
    """
    A generator that generates batches of (arabic numerals, roman numerals) pairs
    """
    _values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    _symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

    def __init__(self, batch_size: int, n_batches: int, max: int=1999):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.max = max

    def __iter__(self):
        for _ in range(self.n_batches):
            input_strings, target_strings = self.generate(self.batch_size)
            yield input_strings, target_strings

    def generate(self, n: int) -> Tuple[List[str], List[str]]:
        """
        generates 'n' pairs of arabic numeral/roman numeral numbers
        """
        numbers = np.exp(np.random.uniform(0, np.log(self.max), n)).round().astype(int)
        remainder = numbers
        quotients = []
        for v in self._values:
            q, remainder = np.divmod(remainder, v)
            quotients.append(q)
        roman_numerals = ["".join(s*c for s, c in zip(self._symbols, counts))
                          for counts in np.transpose(quotients)]
        arabic_numerals = [str(i) for i in numbers]
        return arabic_numerals, roman_numerals



if __name__ == "__main__":
    import IPython
    gene = RomanNumeralsGenerator(10, 1)
    IPython.embed()
