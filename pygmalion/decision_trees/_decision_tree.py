from typing import List, Set, Iterable, Optional, Union
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from ._branch import Branch
from pygmalion._model import Model


DATAFRAME_LIKE = Union[pd.DataFrame, dict, Iterable]


class DecisionTree(Model):

    def __repr__(self):
        max_depth = max(leaf.depth for leaf in self.leafs)
        n_leafs = len(self.leafs)
        return type(self).__name__+f"(target={self.target}, inputs={self.inputs}, n_leafs={n_leafs}, max_depth={max_depth})"

    def __init__(self, inputs: List[str], target: str):
        self.n_observations = None
        self.leafs: Set[Branch] = set()
        self.root = None
        self.inputs = inputs
        self.target = target

    def evaluator(self, target: torch.Tensor):
        """
        From a torch.Tensor of target values returns the model prediction for the given leaf
        """
        raise NotImplementedError()

    def gain(self, target: torch.Tensor, split: torch.Tensor):
        """
        Returns the gain associated to each split.
        Can be set to -inf to ignore a split.

        Parameters
        ----------
        target : torch.Tensor
            tensor of shape (n_observations)
        split : torch.Tensor
            tensor of shape (n_observations, n_splits)
        
        Returns
        -------
        torch.Tensor :
            tensor of losses of shape (n_splits)
        """
        raise NotImplementedError()
    
    def target_preprocessor(self, data: pd.Series, dtype: np.dtype) -> torch.Tensor:
        """
        Converts the pd.Series into a torch.Tensor
        """
        return torch.from_numpy(data.to_numpy(dtype=dtype))

    def fit(self, df: pd.DataFrame, target: Optional[pd.Series]=None, max_depth: Optional[int]=None, min_leaf_size: int=1,
            max_leaf_count: Optional[int]=None, device: torch.device="cpu", dtype=np.float64) -> str:
        """
        Fit the decision tree to observations

        Parameters
        ----------
        df : pd.DataFrame
            the dataframe to fit on
        target : pd.Series or None
            the target if it should not be read from the dataframe
        max_depth : int or None
            the maximum depth of the tree
        min_leaf_size : int
            minimum number of observations in a split for the split to be valid
        max_leaf_count : int or None
            the number of leafs before fitting stops (each split creates one additional)
        device : torch.device
            the device on which to perform the best split search
        """
        self.n_observations = len(df)
        inputs = [torch.from_numpy(df[col].to_numpy(dtype=dtype)).to(device) for col in self.inputs]
        target = self.target_preprocessor(df[self.target] if target is None else target, dtype).to(device)
        self.root = Branch(inputs=inputs, target=target, variables=self.inputs,
                           depth=0, max_depth=max_depth, min_leaf_size=min_leaf_size,
                           gain=self.gain, evaluator=self.evaluator)
        self.leafs = {self.root}
        while True:
            if (max_leaf_count is not None) and (len(self.leafs) >= max_leaf_count):
                break
            splitable_leafs = [leaf for leaf in self.leafs if leaf.is_splitable]
            if len(splitable_leafs) == 0:
                break
            splited = max(splitable_leafs, key=lambda x: x.gain)
            self.leafs.remove(splited)
            splited.grow()
            self.leafs.add(splited.inferior_or_equal)
            self.leafs.add(splited.superior)
        for leaf in self.leafs:
            if leaf.is_splitable:
                leaf._clean()
    
    def predict(self, df: DATAFRAME_LIKE) -> np.ndarray:
        """
        make a prediction
        """
        if self.root is None:
            raise RuntimeError("Cannot evaluate model before it was fited")
        df = self._as_dataframe(df)
        self.root.propagate(df.reset_index(drop=True)[self.inputs])
        result = np.full((len(df),), float("nan"), dtype=np.float64)
        for leaf in self.leafs:
            sub = leaf._df.index
            result[sub] = leaf.value
            leaf._df = None
        return result
    
    @property
    def branches(self) -> Iterable[Branch]:
        """
        Returns all the branches of the decision tree
        """
        if self.root is None:
            return (b for b in [])
        else:
            return (b for iterable in ([self.root], self.root.childs) for b in iterable)

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "branches": self.root.dump,
                "inputs": list(self.inputs),
                "target": self.target}

    @classmethod
    def from_dump(cls, dump: dict) -> "DecisionTree":
        obj = cls.__new__(cls)
        obj.root = Branch.from_dump(dump["branches"])
        obj.inputs = dump["inputs"]
        obj.target = dump["target"]
        obj.leafs = set(b for b in obj.branches if b.is_leaf)
        return obj
    
    def _as_dataframe(self, data: DATAFRAME_LIKE) -> pd.DataFrame:
        """
        Converts any ill formated input into a DataFrame
        """
        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
        elif not isinstance(data, pd.DataFrame):
            data = np.array(data)
            if len(data.shape) == 1:
                data = data[None, ...]
            data = pd.DataFrame(data=data, columns=self.inputs)
        return data


class DecisionTreeRegressor(DecisionTree):

    def evaluator(self, target: torch.Tensor):
        """
        From a torch.Tensor of target values returns the model prediction for the given leaf
        """
        return target.mean().cpu().item()

    def gain(self, target: torch.Tensor, splits: torch.Tensor):
        """
        Returns the MSE (Mean Squared Error) gain associated to each split

        Parameters
        ----------
        target : torch.Tensor
            tensor of shape (n_observations)
        split : torch.Tensor
            tensor of shape (n_splits, n_observations)
        
        Returns
        -------
        torch.Tensor :
            tensor of losses of shape (n_splits)
        """
        pred = target.mean()
        SSE = ((target - pred)**2).sum()
        pred_left, pred_right = (target.unsqueeze(0) * splits).sum(dim=1) / splits.sum(dim=1), (target.unsqueeze(0) * ~splits).sum(dim=1) / (~splits).sum(dim=1)
        SSE_left, SSE_right = ((target.unsqueeze(0) - pred_left.unsqueeze(1))**2 * splits).sum(dim=1), ((target.unsqueeze(0) - pred_right.unsqueeze(1))**2 * ~splits).sum(dim=1)
        return (SSE - SSE_left - SSE_right) / self.n_observations


class DecisionTreeClassifier(DecisionTree):

    def target_preprocessor(self, data: pd.Series, dtype: np.dtype) -> torch.Tensor:
        """
        Converts the pd.Series into a torch.Tensor
        """
        return torch.tensor([self._class_to_index[c] for c in data], dtype=torch.long)

    def evaluator(self, series: torch.Tensor):
        """
        From a torch.Tensor of target values returns the model prediction for the given leaf
        """
        return series.mode().values.cpu().item()

    def gain(self, target: torch.Tensor, splits: torch.Tensor):
        """
        Returns the Gini gain associated to each split

        Parameters
        ----------
        target : torch.Tensor
            tensor of shape (n_observations)
        split : torch.Tensor
            tensor of shape (n_splits, n_observations)
        
        Returns
        -------
        torch.Tensor :
            tensor of gains of shape (n_splits)
        """
        n_splits, n_obs = splits.shape
        classes = F.one_hot(target)
        p = classes.sum(dim=0) / n_obs
        gini = (p * (1-p)).mean()
        n_left = splits.sum(dim=1)
        n_right = n_obs - n_left
        count_left = (classes.unsqueeze(0) * splits.long().unsqueeze(-1)).sum(dim=1)
        count_right = (classes.unsqueeze(0) * (~splits).long().unsqueeze(-1)).sum(dim=1)
        p_left, p_right = count_left / n_left.unsqueeze(-1), count_right / n_right.unsqueeze(-1)
        gini_left, gini_right = (p_left * (1 - p_left)).mean(dim=-1), (p_right * (1 - p_right)).mean(dim=-1)
        return (gini*n_obs - gini_left*n_left - gini_right*n_right) / self.n_observations

    def __init__(self, inputs: List[str], target: str, classes: List[str]):
        super().__init__(inputs, target)
        self.classes = classes
        self._class_to_index = {c: i for i, c in enumerate(classes)}

    def predict(self, df: DATAFRAME_LIKE, indexes: bool=False) -> Union[List[str], np.ndarray]:
        """
        make a prediction

        Parameters
        ----------
        df : pd.dataFrame
            inputs to predict on
        indexes : bool
            if True return indexes of predicted classes instead of list of class names
        """
        if self.root is None:
            raise RuntimeError("Cannot evaluate model before it was fited")
        df = self._as_dataframe(df)
        self.root.propagate(df.reset_index(drop=True)[self.inputs])
        result = np.array([None]*len(df))
        for leaf in self.leafs:
            sub = leaf._df.index
            result[sub] = leaf.value
            leaf._df = None
        if not indexes:
            result = [self.classes[i] for i in result]
        return result