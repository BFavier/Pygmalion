from typing import List, Dict, Iterable, Optional, Union
from itertools import repeat
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from ._decision_tree import DecisionTreeRegressor, DATAFRAME_LIKE, MONOTONICITY
from pygmalion._model import Model


class GradientBoostingRegressor(Model):

    def __repr__(self) -> str:
        return type(self).__name__+f"(target={self.target}, inputs={self.inputs}, n_trees={len(self.trees)})"

    def __init__(self, inputs: List[str], target: str, monotonicity_constraints: Dict[str, MONOTONICITY]={}):
        """
        """
        self.inputs = inputs
        self.target = target
        self.trees : List[DecisionTreeRegressor] = []
        self.monotonicity_constraints = monotonicity_constraints

    def fit(self, data: Union[pd.DataFrame, Iterable[pd.DataFrame]],
            n_trees: int=100, learning_rate: float=0.1,
            max_depth: Optional[int]=None, min_leaf_size: int=1,
            max_leaf_count: Optional[int]=None,
            verbose: bool=True, device: torch.device="cpu"):
        """
        """
        if isinstance(data, pd.DataFrame):
            data = repeat(data)
            single_batch = True
        else:
            single_batch = False
        if verbose:
            data = tqdm(data, total=n_trees)
        try:
            for i, df in enumerate(data):
                if i >= n_trees:
                    break
                if single_batch and i > 0:
                    residual -= lr * tree.predict(df)
                else:
                    residual = df[self.target] - self.predict(df)
                lr = 1.0 if len(self.trees) == 0 else learning_rate
                md = 0 if len(self.trees) == 0 else max_depth
                tree = DecisionTreeRegressor(self.inputs, self.target, self.monotonicity_constraints)
                tree.fit(df, residual, max_depth=md, min_leaf_size=min_leaf_size,
                         max_leaf_count=max_leaf_count, device=device)
                self.trees.append((lr, tree))
                if verbose:
                    RMSE = np.mean(df[self.target]**2)**0.5
                    data.set_postfix(**{"train RMSE": f"{RMSE:.3g}"})
        except KeyboardInterrupt:
            pass

    def predict(self, df: DATAFRAME_LIKE) -> np.ndarray:
        """
        Returns the prediction of the model
        """
        res = np.zeros(len(df))
        for res in self.predict_partial(df):
            pass
        return res

    def predict_partial(self, df: DATAFRAME_LIKE) -> Iterable[np.ndarray]:
        """
        Predict the target after each tree is succesively applied
        """
        df = DecisionTreeRegressor._as_dataframe(self, df)
        predicted = np.zeros(len(df))
        for lr, tree in self.trees:
            predicted = predicted + lr * tree.predict(df)
            yield predicted

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "inputs": list(self.inputs),
                "target": self.target,
                "trees": [[lr, tree.dump] for lr, tree in self.trees],
                "monotonicity_constraints": self.monotonicity_constraints}

    @classmethod
    def from_dump(cls, dump: dict) -> "GradientBoostingRegressor":
        obj = cls.__new__(cls)
        obj.trees = [(lr, DecisionTreeRegressor.from_dump(tree)) for lr, tree in dump["trees"]]
        obj.inputs = dump["inputs"]
        obj.target = dump["target"]
        obj.monotonicity_constraints = {k: MONOTONICITY(v) for k, v in dump["monotonicity_constraints"].items()}
        return obj

    @property
    def feature_importances(self) -> dict:
        """
        Returns a dictionnary of feature importance for each input feature
        """
        fis = [(lr, tree.feature_importances) for lr, tree in self.trees]
        fi = {k: sum(_fi.get(k, 0.) * lr for lr, _fi in fis) for k in self.inputs}
        fi = {k: v for k, v in sorted(fi.items(), key=lambda item: item[1], reverse=True)}
        return fi