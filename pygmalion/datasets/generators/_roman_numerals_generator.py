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
        # numbers = np.exp(np.random.uniform(0, np.log(self.max), n)).round().astype(int)
        if n < self.max:
            numbers = np.random.permutation(self.max)[:n]
        else:
            numbers = np.random.randint(0, self.max+1, n)
        remainder = numbers
        quotients = []
        for v in self._values:
            q, remainder = np.divmod(remainder, v)
            quotients.append(q)
        roman_numerals = ["".join(s*c for s, c in zip(self._symbols, counts))
                          for counts in np.transpose(quotients)]
        arabic_numerals = [str(i) for i in numbers]
        return arabic_numerals, roman_numerals

    @classmethod
    def arabic_to_roman(cls, arabic_number: int) -> str:
        """
        converts an arabic number to roman numbers
        """
        roman_number = ""
        for v, s in zip(cls._values, cls._symbols):
            roman_number += s*(arabic_number // v)
            arabic_number = arabic_number % v
        return roman_number

    @classmethod
    def roman_to_arabic(cls, roman_number: str) -> int:
        """
        converts a roman number to arabic
        """
        arabic_number = 0
        for v, s in zip(cls._values, cls._symbols):
            while roman_number.startswith(s):
                roman_number = roman_number[len(s):]
                arabic_number += v
        return arabic_number


if __name__ == "__main__":
    import IPython
    gene = RomanNumeralsGenerator(10, 1)
    IPython.embed()
