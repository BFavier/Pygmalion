import pandas as pd
import torch
from typing import List, Callable


class Branch:

    def __init__(self, depth: int, device: torch.device, max_depth: int, min_leaf_size: int):
        self.device = device
        self.variable, self.threshold, self.criterion = self._best_split() if depth < max_depth else (None,)*3
        self.inferior_or_equal, self.superior = None, None

    def _best_split(self, df: pd.DataFrame, input_columns: List[str], target: str, criterion: Callable, min_leaf_size: int):
        """
        Of all possible splits of the data, gets the best split
        """
        inputs = [torch.from_numpy(df[col].to_numpy()).to(self.device) for col in input_columns]
        target = torch.from_numpy(df[target].to_numpy()).to(self.device)
        uniques = (X.unique(sorted=True) for X in inputs)
        non_nan = (X[~torch.isnan(X)] for X in uniques)
        low_high = ((X, torch.cat([X[1:], X[-1:]], dim=0)) for X in non_nan)
        boundaries = [(0.5*low + 0.5*high) for low, high in low_high]
        all_splits = [X.reshape(-1, 1) > b.reshape(1, -1) for X, b in zip(inputs, boundaries)]
        scores = [torch.min(
                            torch.masked_fill(criterion(target, splits).to("cpu"),
                                              ((splits.sum(dim=-1) < min_leaf_size) | (~splits.sum(dim=-1) < min_leaf_size)).unsqueeze(-1),
                                              float("inf")),
                            dim=0)
                  for splits in all_splits]
        col, (crit, i) = min(enumerate(scores), key=lambda x: x[1].values)
        crit = crit.item()
        if not torch.isfinite(crit):
            return None, None, None
        return input_columns[col], boundaries[col][i].item(), crit

