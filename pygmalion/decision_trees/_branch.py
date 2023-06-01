import pandas as pd
import numpy as np
import torch
from typing import List, Callable, Optional


class Branch:

    @classmethod
    def from_dump(cls, dump: dict) -> "Branch":
        obj = cls.__new__()
        obj.n_observations = dump["n_observations"]
        obj.depth = dump["depth"]
        obj.value = dump["value"]
        obj.variable = dump["variable"]
        obj.threshold = dump["threshold"]
        obj.gain = dump["gain"]
        obj.inferior_or_equal = dump["inferior_or_equal"]
        if isinstance(obj.inferior_or_equal, dict):
            obj.inferior_or_equal = cls.from_dump(obj.inferior_or_equal)
        obj.superior = dump["superior"]
        if isinstance(obj.superior, dict):
            obj.superior = cls.from_dump(obj.superior)
        return obj

    def __repr__(self):
        if self.is_leaf:
            gain = None if self.gain is None else f"{self.gain:.3g}"
            return f"Branch(value={self.value:.3g}, depth={self.depth}, n_observations={self.n_observations:.3g}, gain={gain})"
        else:
            return f"Branch(variable={self.variable}, threshold={self.threshold:.3g}, gain={self.gain:.3g})"

    def __init__(self, df: pd.DataFrame, input_columns: List[str], target: str,
                 max_depth: Optional[int], min_leaf_size: int, target_preprocessor: Callable,
                 gain: Callable, evaluator: Callable, depth: int, device: torch.device):
        self._df = df
        self._input_columns, self._target = input_columns, target
        self._target_preprocessor = target_preprocessor
        self._gain = gain
        self._evaluator = evaluator
        self._max_depth = max_depth
        self._min_leaf_size = min_leaf_size
        self._device = device
        self.n_observations = len(df)
        self.depth = depth
        self.value = evaluator(df[target])
        if max_depth is None or (depth < max_depth):
            self.variable, self.threshold, self.gain = self._best_split()
        else:
            self.variable, self.threshold, self.gain = None, None, None
        self.inferior_or_equal, self.superior = None, None
        if not self.is_splitable:
            del self._df

    def _best_split(self):
        """
        Of all possible splits of the data, gets the best split
        """
        inputs = [torch.from_numpy(self._df[col].to_numpy(dtype=np.float32)).to(self._device) for col in self._input_columns]
        target = self._target_preprocessor(self._df[self._target]).to(self._device)
        uniques = (X.unique(sorted=True) for X in inputs)
        non_nan = (X[~torch.isnan(X)] for X in uniques)
        inf = torch.full((1,), float("inf"), dtype=torch.float32, device=self._device)
        low_high = ((X, torch.cat([X[1:], inf], dim=0)) for X in non_nan)
        boundaries = [(0.5*low + 0.5*high) for low, high in low_high]
        all_splits = [X.reshape(-1, 1) <= b.reshape(1, -1) for X, b in zip(inputs, boundaries)]
        gains = [torch.max(
                            torch.masked_fill(self._gain(target, splits).cpu(),
                                              ((splits.sum(dim=0).cpu() < self._min_leaf_size) | ((~splits).sum(dim=0).cpu() < self._min_leaf_size)),
                                              -float("inf")),
                            dim=0)
                  for splits in all_splits]
        col, (gain, i) = max(enumerate(gains), key=lambda x: x[1].values)
        if not torch.isfinite(gain):
            return None, None, None
        gain = gain.item()
        return self._input_columns[col], boundaries[col][i].item(), gain

    def grow(self):
        """
        grows the branch by creating two sub-branches
        """
        if not self.is_leaf:
            raise RuntimeError("Cannot grow an already grown non-leaf Branch")
        if not self.is_splitable:
            raise ValueError("Cannot grow a non-splitable Branch")
        self.inferior_or_equal = Branch(self._df[self._df[self.variable] <= self.threshold],
                                        self._input_columns, self._target,
                                        self._max_depth, self._min_leaf_size,
                                        self._target_preprocessor, self._gain, self._evaluator,
                                        self.depth+1, self._device)
        self.superior = Branch(self._df[self._df[self.variable] > self.threshold],
                               self._input_columns, self._target,
                               self._max_depth, self._min_leaf_size,
                               self._target_preprocessor, self._gain, self._evaluator,
                               self.depth+1, self._device)
        del self._df
    
    def propagate(self, df: pd.DataFrame):
        """
        propagate a dataframe to the subbranches and save subset in leafs
        """
        if self.is_leaf:
            self._df = df
        else:
            self.inferior_or_equal.propagate(df[df[self.variable] <= self.threshold])
            self.superior.propagate(df[df[self.variable] > self.threshold])

    @property
    def is_splitable(self) -> bool:
        """
        returns True if the branch can be further splited
        """
        return (self.variable is not None) and (self.threshold is not None)

    @property
    def is_leaf(self) -> bool:
        """
        Returns true if the branch is not splited further (yet)
        """
        return (self.inferior_or_equal is None) or (self.superior is None)
    
    @property
    def dump(self) -> dict:
        return {"n_observations": self.n_observations,
                "depth": self.depth,
                "value": self.value,
                "variable": self.variable,
                "threshold": self.threshold,
                "gain": self.gain,
                "inferior_or_equal": None if self.inferior_or_equal is None else self.inferior_or_equal.dump,
                "superior": None if self.superior is None else self.superior.dump}

