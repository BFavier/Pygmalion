import pandas as _pd
import numpy as _np
from ._decision_tree import RegressionTree as _RegressionTree
from machine_learning._templates import RegressorTemplate as _RegressorTemplate
from machine_learning._templates import ClassifierTemplate as _ClassifierTemplate
import machine_learning._utilities as _utils


class GradientBoostedRegressionTree(_RegressorTemplate):
    """Gradient Boosted Regression Tree"""

    def __init__(self):
        self.shrinkage = 0.1
        self.trees = []

    def _MSE(self, predicted: _np.ndarray, target: _np.ndarray) -> float:
        """MSE loss function"""
        return _np.mean((predicted-target)**2)

    def _grad(self, predicted: _np.ndarray,
              target: _np.ndarray) -> _np.ndarray:
        """Gradient of the MSE loss function:
        dLoss/df = d/df(sum([yi-fi]²)/N) = 2*sum(yi-fi)/N
        => dLoss/df ~ sum(yi - fi)/N
        """
        return predicted - target

    def fit(self, x: _pd.DataFrame, y: _np.ndarray, validation_data=None,
            validation_fraction=0.2, n_max_trees=100, shrinkage=0.1,
            patience=10, verbose=True, **kwargs):
        """
        Fit the model on the given data
        """
        # Getting some validation data
        if validation_data is None:
            data = _utils.train_test(x, y, validation_fraction)
            x_train, y_train, x_val, y_val = data
        else:
            x_train, y_train = x, y
            x_val, y_val = validation_data
        # initializing variables
        self.shrinkage = shrinkage
        self.trees = []
        target = y_train
        predicted_train = _np.zeros(y_train.shape)
        predicted_val = _np.zeros(y_val.shape)
        training_loss = []
        validation_loss = []
        best_loss = 0
        if verbose:
            print("Fitting gradient boosted regression tree:")
        # Training loop
        try:
            for n_trees in range(n_max_trees):
                tree = _RegressionTree()
                tree.fit(x_train, target, **kwargs)
                predicted_train += self.shrinkage*tree.predict(x_train)
                predicted_val += self.shrinkage*tree.predict(x_val)
                self.trees.append(tree)
                train_loss = self._MSE(predicted_train, y_train)
                training_loss.append(train_loss)
                val_loss = self._MSE(predicted_val, y_val)
                validation_loss.append(val_loss)
                target = -self._grad(predicted_train, y_train)
                if verbose:
                    print(f"\ttraining loss={train_loss:.3g}"
                          f"\tvalidation loss={val_loss:.3g}")
                if val_loss <= validation_loss[best_loss]:
                    best_loss = n_trees
                else:
                    if (n_trees - best_loss) > patience:
                        break
        except KeyboardInterrupt:
            pass
        # Removing excess DecisionTrees
        self.trees = self.trees[:best_loss+1]

    def predict(self, x: _pd.DataFrame):
        """
        predict the model
        """
        s = self.shrinkage
        return s*_np.sum([tree.predict(x) for tree in self.trees], axis=0)


class GradientBoostedClassificationTree(_ClassifierTemplate):
    """Gradient Boosted Classification Tree"""

    def __init__(self):
        self.shrinkage = 0.1
        self.trees = dict()

    def _softmax(self, output: _pd.DataFrame) -> _pd.DataFrame:
        """
        Calculate the softmax of the model's output.
        This converts the unborned model outputs for each categories
        to a probability 0 < pj < 1, such that sum_j(p_j) = 1.
            pij = exp(f[i,j]) / sum_j(exp(f[i,j]))
        with i the index of the observations,
        and j the index of the categories to predict,
        f[i,j] is the model's output at observation i for category j
        """
        exp = _np.exp(output.values)
        denominator = _np.sum(exp, axis=1)
        exp /= denominator[:, None]
        return _pd.DataFrame(data=exp, columns=self.categories)

    def _cross_entropy(cls, proba_predicted: _pd.DataFrame,
                       proba_target: _pd.DataFrame) -> float:
        """
        calculate the cross entropy between predicted/target probabilities.
            -sum_i,j(c[i,j] * ln(p[i,j]))
        which equals to:
            -sum_i(ln(p[i,k])
        where:
        i is the index over the observations,
        j is the index over the categories to predict,
        k is the target category of the ith observation,
        c[i,j] is the one-hot target vector (1 if j==k, 0 otherwise),
        p[i,j] is the predicted probability of category j for observation i
        """
        return -_np.mean(proba_target.values * _np.log(proba_predicted))

    def _log_likelihood(self, output: _pd.DataFrame,
                        target: _pd.DataFrame) -> float:
        """
        Returns the log-likelihood loss function from the non-normalized
        predictions and the target categories.
        It is calculated as the cross entropy of the softmaxed model's output.
        This reduces to:
            L = - sum_i,j( c[i,j] * ln( exp(f[i,j]) / sum_j(exp(f[i,j])) ) )
        which equals to:
            L = - sum_i(ln( exp(f[i,k]) / sum_j(exp(f[i,j]))))
        where:
        i is the index over the observations,
        j is the index over the categories to predict,
        k is the target category of the ith observation,
        c[i,j] is the one-hot target vector (1 if j==k, 0 otherwise),
        f[i,j] is the predicted value for observation i, class j
        (before the softmax is applied)
        """
        proba_predicted = self._softmax(output).values
        return self._cross_entropy(proba_predicted, target)

    def _grad(self, output: _pd.DataFrame,
              target: _pd.DataFrame) -> _pd.DataFrame:
        """
        returns the gradient (the Jacobian actually) of the loss function
        (the log-likelihood) relative to the model's prediction.
        It reduces to:
            dL/df[i,j] = p[i,j] - c[i,j]
        """
        proba_predicted = self._softmax(output)
        return proba_predicted - target

    def _to_categories(self, one_hot: _pd.DataFrame) -> _np.ndarray:
        """
        Converts a dataframe of one-hot encoded vectors to a list of
        categories.
        """
        return [self.categories[i] for i in _np.argmax(one_hot.values, axis=1)]

    def _to_one_hot(self, y: _np.ndarray) -> _pd.DataFrame:
        """
        Converts a vector of observed categories to a dataframe of one-hot
        encoded vectors.
        Each column of the dataframe is a category, each line corresponds to
        an observation.
        """
        y = _np.array(y)  # in case a _pd.Series is passed instead
        data = [[1. if c == y[j] else 0 for c in self.categories]
                for j in range(len(y))]
        return _pd.DataFrame(data=data, columns=self.categories)

    def _scalar_output(self, x: _pd.DataFrame, category: str):
        """
        Calculate the raw scalar output for a given category,
        for all observations x
        """
        s = self.shrinkage
        c = category
        return s*_np.sum([tree.predict(x) for tree in self.trees[c]], axis=0)

    def _output(self, x: _pd.DataFrame) -> _pd.DataFrame:
        """
        Calculate the raw output of the model without converting back
        to categories nor probabilities
        """
        vectorial = _np.transpose([self._scalar_output(x, c)
                                  for c in self.categories])
        return _pd.DataFrame(data=vectorial, columns=self.categories,
                             dtype=float)

    def fit(self, x: _pd.DataFrame, y: _np.ndarray, validation_data=None,
            validation_fraction=0.2, n_max_trees=100, shrinkage=0.1,
            patience=10, verbose=True, **kwargs):
        """
        Fit the model on the given data
        """
        # Getting some validation data
        if validation_data is None:
            data = _utils.train_test(x, y, validation_fraction)
            x_train, y_train, x_val, y_val = data
        else:
            x_train, y_train = x, y
            x_val, y_val = validation_data
        # initializing variables
        self.categories = _np.unique(_np.concatenate([y_train, y_val]))
        self.trees = {c: [] for c in self.categories}
        val_target = self._to_one_hot(y_val)
        train_target = self._to_one_hot(y_train)
        target = train_target
        training_loss = []
        validation_loss = []
        best_loss = 0
        if verbose:
            print("Fitting gradient boosted classification tree:")
        # Training loop
        try:
            for n_trees in range(n_max_trees):
                for c in self.categories:
                    tree = _RegressionTree()
                    tree.fit(x_train, target[c], **kwargs)
                    self.trees[c].append(tree)
                    if verbose:
                        print(".", end="", flush=True)
                output = self._output(x_train)
                train_loss = self._log_likelihood(output, train_target)
                training_loss.append(train_loss)
                val_loss = self._log_likelihood(self._output(x_val),
                                                val_target)
                validation_loss.append(val_loss)
                target = -self._grad(output, train_target)
                if verbose:
                    print(f"\ttraining loss={train_loss:.3g}"
                          f"\tvalidation loss={val_loss:.3g}")
                if val_loss <= validation_loss[best_loss]:
                    best_loss = n_trees
                else:
                    if (n_trees - best_loss) > patience:
                        break
        except KeyboardInterrupt:
            pass
        # Removing excess DecisionTrees
        for c in self.categories:
            self.trees[c] = self.trees[c][:best_loss+1]

    def predict(self, x: _pd.DataFrame) -> _np.ndarray:
        """
        Predict the model on the given data
        """
        return self._to_categories(self._output(x))
