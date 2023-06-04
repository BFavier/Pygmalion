from typing import List, Iterable, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from ._decision_tree import DecisionTreeRegressor


class GradientBoostingRegressor:

    def __init__(self, inputs: List[str], target: str):
        """
        """
        self.inputs = inputs
        self.target = target
        self.trees = []
    
    def fit(self, df: pd.DataFrame, n_trees: int, learning_rate: float = 0.3,
            max_depth: Optional[int]=None, min_leaf_size: int=1,
            max_leaf_count: Optional[int]=None, verbose: bool=True,
            device: torch.device="cpu"):
        """
        """
        df = df.copy()
        counter = range(n_trees)
        if verbose:
            counter = tqdm(counter)
        try:
            for _ in counter:
                lr = 1.0 if len(self.trees) == 0 else learning_rate
                md = 0 if len(self.trees) == 0 else max_depth
                tree = DecisionTreeRegressor(self.inputs, self.target)
                tree.fit(df, max_depth=md, min_leaf_size=min_leaf_size,
                         max_leaf_count=max_leaf_count, device=device)
                self.trees.append((lr, tree))
                df[self.target] -= lr * tree.predict(df)
                if verbose:
                    RMSE = np.mean(df[self.target]**2)**0.5
                    counter.set_postfix(**{"RMSE": f"{RMSE:.3g}"})
        except KeyboardInterrupt:
            pass

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns the prediction of the model
        """
        for res in self.predict_partial(df):
            pass
        return res

    def predict_partial(self, df: pd.DataFrame) -> Iterable[np.ndarray]:
        """
        Predict the target after each tree is succesively applied
        """
        df = DecisionTreeRegressor._as_dataframe(self, df)
        predicted = np.zeros(len(df))
        for lr, tree in self.trees:
            predicted = predicted + lr * tree.predict(df)
            yield predicted