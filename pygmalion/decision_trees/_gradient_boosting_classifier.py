from ._decision_tree import DecisionTreeRegressor, DATAFRAME_LIKE
from typing import List, Iterable, Optional, Union
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


class GradientBoostingClassifier:

    def __init__(self, inputs: List[str], target: str, classes: list):
        self.inputs = inputs
        self.target = target
        self.classes = classes
        self._class_to_index = {c: i for i, c in enumerate(classes)}

    def fit(self, df: pd.DataFrame, n_trees: int, learning_rate: float = 0.3,
            max_depth: Optional[int]=None, min_leaf_size: int=1,
            max_leaf_count: Optional[int]=None, verbose: bool=True,
            device: torch.device="cpu", dtype: np.dtype=np.float64):
        self.trees = []
        frequencies = df[self.target].value_counts(normalize=True)
        for c in self.classes:
            if c not in frequencies.index:
                raise ValueError(f"Target class '{c}' is not present in the training dataset")
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
                    target = np.repeat(np.array([[np.log(frequencies.get(c, 0.))] for c in self.classes]), len(df), axis=1)
                    for trg in target:
                        tree = DecisionTreeRegressor(self.inputs, self.target)
                        tree.fit(df, pd.Series(trg), max_depth=0, device=device, dtype=dtype)
                        trees.append(tree)
                    self.trees.append((1.0, trees))
                else:
                    denominator = (1 / np.exp(predicted).sum(axis=0))
                    for pred, kronecker in zip(predicted, class_mask):
                        tree = DecisionTreeRegressor(self.inputs, self.target)
                        trg = kronecker - pred/denominator
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

    def _predicted(self, df: DATAFRAME_LIKE) -> np.ndarray:
        predicted = np.zeros((len(self.classes), len(df)))
        for lr, trees in self.trees:
            predicted += lr * np.stack([tree.predict(df) for tree in trees], axis=0)
            yield predicted

    def predict(self, df: DATAFRAME_LIKE, probabilites: bool=False, index: bool=False) -> Union[pd.DataFrame, np.ndarray, List[str]]:
        """
        Returns the prediction of the model
        """
        for res in self._predicted(df):
            pass
        if probabilites:
                return pd.DataFrame(data=np.transpose(res), columns=self.inputs)
        elif index:
            return np.argmax(res, axis=0)
        else:
            return [self.classes[c] for c in np.argmax(res, axis=0)]

    def predict_partial(self, df: DATAFRAME_LIKE, probabilites: bool=False, index: bool=False) -> Iterable[Union[pd.DataFrame, np.ndarray, List[str]]]:
        """
        Predict the target after each tree is succesively applied
        """
        for predicted in self._predicted(df):
            if probabilites:
                yield pd.DataFrame(data=np.transpose(predicted), columns=self.inputs)
            elif index:
                yield np.argmax(predicted, axis=0)
            else:
                yield [self.classes[c] for c in np.argmax(predicted, axis=0)]

