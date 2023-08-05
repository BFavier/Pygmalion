from ._decision_tree import DecisionTreeRegressor, DATAFRAME_LIKE
from typing import List, Iterable, Optional, Union
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

    def fit(self, df: pd.DataFrame, n_trees: int=100, learning_rate: float=0.1,
            max_depth: Optional[int]=None, min_leaf_size: int=1,
            max_leaf_count: Optional[int]=None, verbose: bool=True,
            device: torch.device="cpu", dtype: np.dtype=np.float64):
        frequencies = df[self.target].value_counts(normalize=True)
        for c in self.classes:
            if c not in frequencies.index:
                warn(f"Target class '{c}' is not present in the training dataset")
        predicted = np.zeros((len(self.classes), len(df)), dtype=dtype)
        class_indexes = np.array([self._class_to_index[c] for c in df[self.target]], dtype=np.uint32)
        observation_indexes = np.arange(len(df))
        class_mask = np.zeros((len(self.classes), len(df)), dtype=np.int32)
        class_mask[class_indexes, observation_indexes] = 1
        counter = range(n_trees)
        if verbose:
            counter = tqdm(counter)
        try:
            for _ in counter:
                trees = []
                if len(self.trees) == 0:
                    target = np.repeat(np.array([[np.log(frequencies.get(c, 1.0E-10))] for c in self.classes]), len(df), axis=1)
                    for trg in target:
                        tree = DecisionTreeRegressor(self.inputs, self.target)
                        tree.fit(df, pd.Series(trg), max_depth=0, device=device, dtype=dtype)
                        trees.append(tree)
                    self.trees.append((1.0, trees))
                else:
                    denominator = (1 / np.exp(predicted).sum(axis=0))
                    for pred, kronecker in zip(predicted, class_mask):
                        tree = DecisionTreeRegressor(self.inputs, self.target)
                        trg = kronecker - 1/denominator
                        tree.fit(df, pd.Series(trg), max_depth=max_depth, min_leaf_size=min_leaf_size, max_leaf_count=max_leaf_count, device=device, dtype=dtype)
                        trees.append(tree)
                    self.trees.append((learning_rate, trees))
                lr, trees = self.trees[-1]
                predicted += lr * np.stack([tree.predict(df) for tree in trees], axis=0)
                if verbose:
                    accuracy = np.mean(predicted.argmax(axis=0) == class_indexes)
                    counter.set_postfix(**{"train accuracy": f"{accuracy:.3%}"})
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