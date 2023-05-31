from typing import List, Set, Optional, Union
import pandas as pd
import torch
from ._branch import Branch


class DecisionTree:

    def __init__(self, inputs: List[str], target: str):
        self._n_observations = None
        self.leafs: Set[Branch] = set()
        self.root = None
        self.inputs = inputs
        self.target = target

    def evaluator(self, series: pd.Series):
        """
        From a pandas series of target values returns the model prediction for the given leaf
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

    def fit(self, df: pd.DataFrame, max_depth: Optional[int]=None, min_leaf_size: int=1, max_leaf_count: Optional[int]=None, device: torch.device="cpu") -> str:
        """
        Fit the decision tree to observations

        Parameters
        ----------
        df : pd.DataFrame
            the dataframe to fit on
        max_depth : int or None
            the maximum depth of the tree
        min_leaf_size : int
            minimum number of observations in a split for the split to be valid
        max_leaf_count : int or None
            the number of leafs before fitting stops (each split creates one additional)
        device : torch.device
            the device on which to perform the best split search
        """
        self._n_observations
        self.root = Branch(df, self.inputs, self.target, max_depth, min_leaf_size, self.gain, self.evaluator, 0, device)
        self.leafs = {self.root}
        while True:
            if (max_leaf_count is not None) and (len(self.leafs) >= max_leaf_count):
                return "max leaf count reached"
            splitable_leafs = [leaf for leaf in self.leafs if leaf.is_splitable]
            if len(splitable_leafs) == 0:
                return "no more splitable leafs"
            splited = max(splitable_leafs, key=lambda x: x.gain)
            self.leafs.pop(splited)
            splited.grow()
    
    def predict(self, df: pd.DataFrame):
        """
        make a prediction
        """
        if self.root is None:
            raise RuntimeError("Cannot evaluate model before it was fited")
        self.root.propagate(df.reset_index(drop=True)[self.inputs])
        ...


class DecisionTreeRegressor(DecisionTree):

    def evaluator(self, series: pd.Series):
        """
        From a pandas series of target values returns the model prediction for the given leaf
        """
        return series.mean()

    def gain(self, target: torch.Tensor, splits: torch.Tensor):
        """
        Returns the MSE (Mean Squared Error) gain associated to each split

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
        mean = target.sum(dim=0)
        var = ((target - mean)**2).sum(dim=0)
        mean_left, mean_right = (target * splits).sum(dim=0), (target * ~splits).sum(dim=0)
        var_left, var_right = ((target - mean_left)**2 * splits).sum(dim=0), ((target - mean_right)**2 * ~splits).sum(dim=0)
        return (var - var_left - var_right) / self._n_observations


class DecisionTreeClassifier(DecisionTree):

    def evaluator(self, series: pd.Series):
        """
        From a pandas series of target values returns the model prediction for the given leaf
        """
        return series.mean()

    def gain(self, target: torch.Tensor, splits: torch.Tensor):
        """
        Returns the MSE (Mean Squared Error) gain associated to each split

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
        mean = target.sum(dim=0)
        var = ((target - mean)**2).sum(dim=0)
        mean_left, mean_right = (target * splits).sum(dim=0), (target * ~splits).sum(dim=0)
        var_left, var_right = ((target - mean_left)**2 * splits).sum(dim=0), ((target - mean_right)**2 * ~splits).sum(dim=0)
        return (var - var_left - var_right) / self._n_observations

    def __init__(self, inputs: List[str], target: str, classes: List[str]):
        super().__init__(inputs, target)
        self.classes = classes