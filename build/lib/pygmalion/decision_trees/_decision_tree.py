import numpy as _np
import pandas as _pd
from machine_learning._templates import RegressorTemplate as _Regressor
from machine_learning._templates import ClassifierTemplate as _Classifier
import machine_learning.library as _lib
from machine_learning.library import decision_trees as _dt


class Branch:
    """ A class representing a branch of a dichotomous tree

    Attributes:
    -----------
    child_true : branch or None
        The child that verifies the condition.
        If None, this branch is a leaf.
    child_false : branch or None
        The child that doesn't verify the condition.
        If None, this branch is a leaf.
    variable : str
        The Name of the variable on which the condition applies.
        This is the left hand sign of the test.
    criterion : str
        The criterion of the test ('<=' or '==')
    threshold : float
        The threshold value for the right hand sign of the test.
    gain : float
        The gain of the split of this branch
    value : object
        For leafs, the value predicted for items that reach the leaf
    mask : np.ndarray
        An array of boolean masking the data that reached this branch
    """

    operation = {"<=": lambda x, y: x <= y,
                 "==": lambda x, y: x == y}

    def __init__(self):
        self.child_true = None
        self.child_false = None
        self.variable = "name"
        self.criterion = "<="
        self.threshold = float("nan")
        self.gain = 0.
        self.value = None
        self.mask = None

    def from_cBranch(self, cbranch: _dt.cBranch, x_num: _pd.DataFrame,
                     x_cat: _pd.DataFrame, y: _np.ndarray):
        """fill branch from the data from a cBranch"""
        if cbranch.is_leaf:
            if (y.dtype == float):
                self.value = cbranch.y_num
            else:
                self.value = y[cbranch.iy_cat]
        else:
            x = x_num if cbranch.split_numerical_x else x_cat
            self.variable = x.columns[cbranch.j]
            self.criterion = "<=" if cbranch.split_numerical_x else "=="
            self.threshold = x[self.variable].iloc[cbranch.i]
            self.gain = cbranch.gain

    def propagate(self, x: _pd.DataFrame, mask: _np.ndarray):
        """Recursively propagate the dataframe in the tree."""
        self.mask = mask
        if not self.is_leaf:
            operator = self.operation[self.criterion]
            test = operator(x[self.variable], self.threshold)
            self.child_true.propagate(x, mask & test)
            self.child_false.propagate(x, mask & ~test)

    def grow(self, ctree: list, index: int, leafs: list,
             x_num: _pd.DataFrame, x_cat: _pd.DataFrame, y: _np.ndarray):
        """Recursively expands the tree to copy the content of the cTree."""
        cbranch = ctree[index]
        self.from_cBranch(cbranch, x_num, x_cat, y)
        if not cbranch.is_leaf:
            self.child_true = Branch()
            self.child_true.grow(ctree, cbranch.child_true, leafs, x_num,
                                 x_cat, y)
            self.child_false = Branch()
            self.child_false.grow(ctree, cbranch.child_false, leafs, x_num,
                                  x_cat, y)
        else:
            leafs.append(self)

    @property
    def is_leaf(self):
        """ True if this branch is a leaf"""
        return (self.child_true is None) and (self.child_false is None)


class DecisionTree:

    def split_num_cat(x: _pd.DataFrame):
        """
        split x into two dataframes:
            - one of numerical data
            - one of categorical data
        """
        is_numerical = [_np.issubdtype(x[c].dtype, _np.number)
                        for c in x.columns]
        c_num = [c for i, c in enumerate(x.columns) if is_numerical[i]]
        c_cat = [c for i, c in enumerate(x.columns) if not is_numerical[i]]
        x_num = x[c_num]
        x_cat = x[c_cat].fillna("")
        return x_num, x_cat

    def __init__(self):
        super().__init__()
        self.root = Branch()
        self.leafs = []

    def grow(self, ctree: list, x_num: _pd.DataFrame, x_cat: _pd.DataFrame,
             y: _np.ndarray):
        """Expand the root of a tree"""
        self.root.grow(ctree, 0, self.leafs, x_num, x_cat, y)

    def predict(self, x: _pd.DataFrame):
        "predict y for the input x"
        if hasattr(self, "classes"):
            L = _np.max([len(s) for s in self.classes])
            y_pred = _np.ndarray(len(x), dtype=f"U{L}")
        else:
            y_pred = _np.ndarray(len(x), dtype=float)
        self.root.propagate(x, _np.array([True]*len(x)))
        for leaf in self.leafs:
            y_pred[leaf.mask] = leaf.value
        return y_pred


class ClassificationTree(DecisionTree, _Classifier):

    def __init__(self):
        super().__init__()

    def fit(self, x: _pd.DataFrame, y: _np.ndarray, **kwargs):
        if isinstance(y, _pd.Series):
            y = y.to_numpy()
        y = y.astype(str)
        self.classes = _np.unique(y)
        x_num, x_cat = DecisionTree.split_num_cat(x)
        ctree = _dt.grow_classifier_tree(x_num.to_numpy(float),
                                         x_cat.to_numpy(str),
                                         y, **kwargs)
        branches = [ctree.branches[i] for i in range(ctree.n_branches)]
        self.grow(branches, x_num, x_cat, y)
        _lib.delete_tree(ctree)


class RegressionTree(DecisionTree, _Regressor):

    def __init__(self):
        super().__init__()

    def fit(self, x: _pd.DataFrame, y: _np.ndarray, **kwargs):
        if isinstance(y, _pd.Series):
            y = y.to_numpy()
        y = y.astype(float)
        assert not _np.isnan(y).any()
        x_num, x_cat = DecisionTree.split_num_cat(x)
        y_pred = _np.ndarray(len(y), float)  # debug
        ctree = _dt.grow_regressor_tree(x_num.to_numpy(float).copy(order='C'),
                                        x_cat.to_numpy(str).copy(order='C'),
                                        y, y_pred=y_pred, **kwargs)
        branches = [ctree.branches[i] for i in range(ctree.n_branches)]
        self.grow(branches, x_num, x_cat, y)
        _dt.delete_tree(ctree)
