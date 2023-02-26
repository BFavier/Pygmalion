from typing import List, Tuple
import operator
import torch
import numpy as np


class OperationsGenerator:
    """
    A generator that generates batches of (operation, result)
    with 'operation' a string representing a numerical operation between two numbers
    and 'result' a string representing the result of the operation
    """

    def __init__(self, tokenizer_in: object, tokenizer_out: object,
                 batch_size: int, n_batches: int,
                 device: torch.device = torch.device("cpu")):
        self.tokenizer_in = tokenizer_in
        self.tokenizer_out = tokenizer_out
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.device = device
        self.operators = [("+", operator.add), ("*", operator.mul), ("-", operator.sub)]

    def __iter__(self):
        for _ in range(self.n_batches):
            input_strings, target_strings = self.generate(self.batch_size)
            input_tensor = torch.tensor([self.tokenizer_in.encode(s) for s in input_strings],
                                        dtype=torch.long, device=self.device)
            target_tensor = torch.tensor([self.tokenizer_out.encode(s) for s in target_strings],
                                         dtype=torch.long, device=self.device)
            yield input_tensor, target_tensor

    def generate(self, n: int) -> Tuple[List[str], List[str]]:
        left, right = np.random.randint(1, 100, n), np.random.randint(1, 100, n)
        operations = np.random.randint(0, len(self.operators), n)
        input_string = [f"{l}{self.operators[i][0]}{r}" for l, i, r in zip(left, operations, right)]
        target_string = [f"{self.operators[i][1](l, r)}" for l, i, r in zip(left, operations, right)]
        return input_string, target_string


if __name__ == "__main__":
    import IPython
    gene = OperationsGenerator(None, 10, 1)
    IPython.embed()
