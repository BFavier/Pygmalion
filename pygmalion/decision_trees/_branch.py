import pandas as pd
import numpy as np
import torch
from typing import List, Iterable, Callable, Optional


class Branch:

    def __repr__(self):
        attrs = ("depth", "n_observations", "value", "variable", "threshold", "gain")
        values = [getattr(self, attr) for attr in attrs]
        reprs = [f"{attr}='{val}" if isinstance(val, str) else f"{attr}={val:.3g}" for attr, val in zip(attrs, values) if val is not None]
        return "Branch("+", ".join(reprs+[f"is_leaf={self.is_leaf}"])+")"

    def __init__(self, inputs: List[torch.Tensor], target: torch.Tensor,
                 variables: List[str], depth: int,
                 max_depth: Optional[int], min_leaf_size: int,
                 gain: Callable, evaluator: Callable):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            tensor of shape (n_observations, n_features)
        target : torch.Tensor
            tensor of shape (n_observations,)
        variables : list of str
            name of each feature to use as input
        depth : int
            current depth of the Branch in the DecisionTree
        max_depth : int or None
            maximum depth of the DecisionTree
        min_leaf_size : int
            minimum number of observations in a leaf for a split to be valid
        gain : Callable
            the gain function used to evaluate splits
        evaluator : Callable
            the evaluation function used to get the prediction for a leaf
        """
        self._inputs = inputs
        self._target = target
        self._variables = variables
        self._gain = gain
        self._evaluator = evaluator
        self._max_depth = max_depth
        self._min_leaf_size = min_leaf_size
        self.n_observations = len(inputs)
        self.depth = depth
        self.value = evaluator(target)
        if max_depth is None or (depth < max_depth):
            self.variable, self.threshold, self.gain = self._best_split()
        else:
            self.variable, self.threshold, self.gain = None, None, None
        self.inferior_or_equal, self.superior = None, None
        if not self.is_splitable:
            self._clean()

    def _best_split(self) -> tuple:
        """
        Of all possible splits of the data, gets the best split.

        Returns
        -------
        tuple :
            The tuple of (variable_index, threshold, gain) achieved with the best split.
            (can be a tuple of None if no valid split were found)
        """
        if len(self._inputs) == 0:
            return (None, None, None)
        uniques = (X.unique(sorted=True) for X in self._inputs)
        non_nan = (X[~torch.isnan(X)] for X in uniques)
        inf = torch.full((1,), float("inf"), dtype=self._inputs[0].dtype, device=self._inputs[0].device)
        low_high = ((X, torch.cat([X[1:], inf], dim=0)) for X in non_nan)
        boundaries = [(0.5*low + 0.5*high) for low, high in low_high]
        all_splits = [X.reshape(1, -1) <= b.reshape(-1, 1) for X, b in zip(self._inputs, boundaries)]
        var_indexes = torch.cat([torch.full((len(S),), i, device=S.device, dtype=torch.long) for i, S in enumerate(all_splits)], dim=0)
        boundaries = torch.cat(boundaries, dim=0)
        all_splits = torch.cat(all_splits, dim=0)
        leaf_size_check = (all_splits.sum(dim=1) >= self._min_leaf_size) & ((~all_splits).sum(dim=1) >= self._min_leaf_size)
        if not leaf_size_check.any():
            return (None, None, None)
        all_splits, var_indexes, boundaries = all_splits[leaf_size_check], var_indexes[leaf_size_check], boundaries[leaf_size_check]
        gain, i = torch.max(self._gain(self._target, all_splits, var_indexes), dim=0)
        if not (gain > -float("inf")):
            return None, None, None
        gain = gain.cpu().item()
        variable = self._variables[var_indexes[i].cpu().item()]
        threshold = boundaries[i].cpu().item()
        return variable, threshold, gain

    def _clean(self):
        """
        Remove useless attributes from the objet after training
        """
        del self._inputs, self._target, self._variables, self._gain, self._evaluator, self._max_depth, self._min_leaf_size

    def grow(self):
        """
        grows the branch by creating two sub-branches
        """
        if not self.is_leaf:
            raise RuntimeError("Cannot grow an already grown non-leaf Branch")
        if not self.is_splitable:
            raise ValueError("Cannot grow a non-splitable Branch")
        inf = (self._inputs[self._variables.index(self.variable)] <= self.threshold)
        self.inferior_or_equal = Branch(inputs=[X[inf] for X in self._inputs], target=self._target[inf],
                                        variables=self._variables, depth=self.depth+1,
                                        max_depth=self._max_depth, min_leaf_size=self._min_leaf_size,
                                        gain=self._gain, evaluator=self._evaluator)
        sup = (self._inputs[self._variables.index(self.variable)] > self.threshold)
        self.superior = Branch(inputs=[X[sup] for X in self._inputs], target=self._target[sup],
                               variables=self._variables, depth=self.depth+1,
                               max_depth=self._max_depth, min_leaf_size=self._min_leaf_size,
                               gain=self._gain, evaluator=self._evaluator)
        self._clean()

    def propagate(self, df: pd.DataFrame):
        """
        propagate recursively a dataframe to the subbranches and save subset in leafs
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
    def childs(self) -> Iterable["Branch"]:
        """
        Recursively returns all the branch below this one
        """
        return (branch for direct_child in (self.inferior_or_equal, self.superior) if direct_child is not None
                for iterable in ([direct_child], direct_child.childs)
                for branch in iterable)

    @classmethod
    def from_dump(cls, dump: dict) -> "Branch":
        obj = cls.__new__(cls)
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

