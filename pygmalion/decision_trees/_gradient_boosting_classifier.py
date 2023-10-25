from ._decision_tree import DecisionTreeRegressor, DATAFRAME_LIKE
from typing import List, Iterable, Optional, Union
from itertools import repeat
from warnings import warn
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pygmalion._model import Model


class GradientBoostingClassifier(Model):

    def __repr__(self):
        return type(self).__name__+f"(target={self.target}, inputs={self.inputs}, classes={self.classes}, n_trees={len(self.trees)})"

    def __init__(self, inputs: List[str], target: str, classes: list):
        self.inputs = inputs
        self.target = target
        self.classes = classes
        self.trees = []
        self._class_to_index = {c: i for i, c in enumerate(classes)}

    def fit(self, data: Union[pd.DataFrame, Iterable[pd.DataFrame]],
            n_trees: int=100, learning_rate: float=0.1,
            max_depth: Optional[int]=None, min_leaf_size: int=1,
            max_leaf_count: Optional[int]=None, verbose: bool=True,
            device: torch.device="cpu", dtype: np.dtype=np.float64):
        if isinstance(data, pd.DataFrame):
            data = repeat(data)
        if verbose:
            data = tqdm(data, total=n_trees)
        try:
            for i, df in enumerate(data):
                if i >= n_trees:
                    break
                trees = []
                class_indexes = np.array([self._class_to_index[c] for c in df[self.target]], dtype=np.uint32)
                if len(self.trees) == 0:
                    predicted = np.zeros((len(df), len(self.classes)))
                    frequencies = df[self.target].value_counts(normalize=True)
                    targets = np.repeat(np.array([[np.log(frequencies.get(c, 1.0E-10))] for c in self.classes]), len(df), axis=1)
                    for trg in targets:
                        tree = DecisionTreeRegressor(self.inputs, self.target)
                        tree.fit(df, pd.Series(trg), max_depth=0, device=device, dtype=dtype)
                        trees.append(tree)
                    self.trees.append((1.0, trees))
                else:
                    class_mask = np.zeros((len(self.classes), len(df)), dtype=np.int32)
                    class_mask[class_indexes, np.arange(len(df))] = 1
                    predicted = self.predict(df, probabilities=True).to_numpy()
                    targets = class_mask * (1 - predicted.T)
                    for trg in targets:
                        tree = DecisionTreeRegressor(self.inputs, self.target)
                        tree.fit(df, pd.Series(trg), max_depth=max_depth, min_leaf_size=min_leaf_size, max_leaf_count=max_leaf_count, device=device, dtype=dtype)
                        trees.append(tree)
                    self.trees.append((learning_rate, trees))
                if verbose:
                    accuracy = np.mean(predicted.argmax(axis=1) == class_indexes)
                    data.set_postfix(**{"train accuracy": f"{accuracy:.3%}"})
        except KeyboardInterrupt:
            pass

    def _predicted(self, df: DATAFRAME_LIKE) -> Iterable[np.ndarray]:
        """
        Returns all individual prediction stages without formating
        """
        predicted = np.zeros((len(self.classes), len(df)))
        for lr, trees in self.trees:
            predicted += lr * np.stack([tree.predict(df) for tree in trees], axis=0)
            yield predicted
        
    def _format_prediction(self, predicted: np.ndarray, probabilities: bool=False, index: bool=False) -> Union[pd.DataFrame, np.ndarray, List[str]]:
        """
        format a prediction of the model
        """
        if probabilities:
                p = np.transpose(np.exp(predicted))
                p /= p.sum(axis=-1)[:, None]
                return pd.DataFrame(data=p, columns=self.classes)
        elif index:
            return np.argmax(predicted, axis=0)
        else:
            return [self.classes[c] for c in np.argmax(predicted, axis=0)]

    def predict(self, df: DATAFRAME_LIKE, probabilities: bool=False, index: bool=False) -> Union[pd.DataFrame, np.ndarray, List[str]]:
        """
        Returns the prediction of the model
        """
        for res in self._predicted(df):
            pass
        return self._format_prediction(res, probabilities, index)

    def predict_partial(self, df: DATAFRAME_LIKE, probabilities: bool=False, index: bool=False) -> Iterable[Union[pd.DataFrame, np.ndarray, List[str]]]:
        """
        Predict the target after each tree is succesively applied
        """
        for predicted in self._predicted(df):
            yield self._format_prediction(predicted, probabilities, index)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "inputs": list(self.inputs),
                "target": self.target,
                "classes": list(self.classes),
                "trees": [[lr, [tree.dump for tree in trees]] for lr, trees in self.trees]}

    @classmethod
    def from_dump(cls, dump: dict) -> "GradientBoostingClassifier":
        obj = cls.__new__(cls)
        obj.trees = [(lr, [DecisionTreeRegressor.from_dump(tree) for tree in trees]) for lr, trees in dump["trees"]]
        obj.inputs = dump["inputs"]
        obj.target = dump["target"]
        obj.classes = dump["classes"]
        return obj