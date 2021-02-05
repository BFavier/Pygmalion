import machine_learning.neural_networks.common as _common
import machine_learning._templates as _templates
import torch as _torch
import types as _types
import machine_learning as _ml
import machine_learning.neural_networks.layers as _layers
import numpy as _np


class Classifier(_common.NNtemplate, _templates.ClassifierTemplate):
    """
    A model that predict scalar values from vectors of observations.

    Attributes:
    -----------
    x : list of str
        name of x observations columns
    classes : list
        classes the inputs can be classified into
    mean : np.ndarray
        The mean value for normalization of input
    std : np.ndarray
        The standard deviation for normalization of input
    dense_layers : list
        list of the dense layers of the model
    args : list or None
        list of arguments used to generate the model
    """

    def __init__(self):
        super().__init__()
        self.dense_layers = []
        self.loss = _torch.nn.functional.cross_entropy
        self.args = None

    def fit(self, observations, labels, layers=(32, 16, 8), validation=None,
            validation_fraction=0.2, restart=True, non_linear="relu",
            l_max=None, **kwargs):
        """
        Fit the model on the given data.

        Parameters:
        -----------
        observations : pd.DataFrame
            real valued observations
        labels : list
            label associated to each observation
        layers : list of int
            The number of neurons in each dense hidden layer
        validation : pandas.DataFrame or None
            data used for validation
        validation_fraction : float
            if validation is None, training data is splited
        restart : bool
            if False, the architecture is not modified
        non_linear : str or function
            non-linear function
        l_max : int or None
            maximum observations in a batch

        Returns:
        --------
        training, validation : pandas.DataFrame

        Other parameters:
        -----------------
        **kwargs from self.train_loop(...)
        """
        training = observations, labels
        # preprocess
        training, validation = self._validation_split(training, validation,
                                                      validation_fraction)
        res = self._preprocess(training, validation)
        self.mean, self.std, self.classes, self.x, _, _ = res
        # create layers
        if restart:
            self._set_layers(layers, non_linear)
        # load data in memory
        train = self._to_tensor(training)
        val = self._to_tensor(validation)
        train, n_training = self._wrap_batch(train, l_max)
        val, n_validation = self._wrap_batch(val, l_max)
        # fitting
        self.train_loop(train, val, n_training, n_validation, **kwargs)
        # returning training and validation data
        return training, validation

    def fit_batch(self, training, validation, x, y, layers,
                  restart=True, non_linear="relu", **kwargs):
        """
        Fit the model on the given batches.

        Parameters:
        -----------
        training : generator function
            function that yields a pandas dataframe
        validation : generator function
            function that yields a pandas dataframe
        x : list of str
            a list of column name for x variables
        y : str
            column name of the prediction
        layers : list of int
            The number of neurons in each dense hidden layer
        restart : bool
            if False, the architecture is not modified
        non_linear : str or function
            non-linear function

        Other parameters:
        -----------------
        **kwargs from self.train_loop(...)
        """
        # preprocess
        res = self._preprocess(training, validation)
        self.mean, self.std, self.classes, self.x, \
            n_training, n_validation = res
        # create layers
        if restart:
            self._set_layers(layers, non_linear)
        # fitting
        self.train_loop(self._wrap(training), self._wrap(validation),
                        n_training, n_validation, **kwargs)

    def predict(self, observations):
        """
        Make a prediction using the trained model.

        Parameters:
        -----------
        observations : pandas.DataFrame

        Returns:
        --------
        labels : list
        """
        X = observations[self.x].values
        tensor = self._array_to_tensor(X, self.mean, self.std)
        with _torch.no_grad():
            output = self(tensor)
        labels = self._tensor_to_labels(output)
        return labels

    def forward(self, tensor):
        """Don't use this"""
        for i, dense in enumerate(self.dense_layers):
            tensor = dense(tensor)
        return tensor

    def _to_tensor(self, data):
        """Converts a dataframe into a tuple of tensors"""
        X, Y = data
        X = X[self.x].values
        X = self._array_to_tensor(X, mean=self.mean, std=self.std)
        Y = self._labels_to_tensor(Y)
        return X, Y

    def _preprocess(self, training, validation):
        """preprocess the data"""
        # batchify if needed
        if not isinstance(training, _types.FunctionType):
            return self._preprocess(lambda: (yield training),
                                    lambda: (yield validation))
        # treat by batches
        sum_, sum2, n = 0., 0., 0.
        classes = set()
        x = None
        # treat training data
        for i, (obs, labels) in enumerate(training()):
            if x is None:
                x = list(obs.columns)
            for key in obs.columns:
                if obs[key].isna().any():
                    raise ValueError(f"At batch {i}: Found NaN in "
                                     f"training x['{key}']")
            sum_, sum2, n = self._sums(obs.values, sum_, sum2, n)
            classes = classes | set(labels)
        n_training = i+1
        # treat validation data
        for i, (obs, labels) in enumerate(validation()):
            for key in obs.columns:
                if obs[key].isna().any():
                    raise ValueError(f"At batch {i}: Found NaN in "
                                     f"validation x['{key}']")
            sum_, sum2, n = self._sums(obs.values, sum_, sum2, n)
            classes = classes | set(labels)
        n_validation = i+1
        # returns values
        mean, std = self._mean_std(sum_, sum2, n)
        return mean, std, list(classes), x, n_training, n_validation

    def _set_layers(self, *args):
        """Remove the previous layers and create new ones"""
        layers, non_linear = args
        self.args = args
        # Deleting previous layers
        for i, _ in enumerate(self.dense_layers):
            delattr(self, f"dense{i}")
        self.dense_layers = []
        # Set dense layers
        dens = _layers.DenseLayer
        n = len(self.x)
        for i, n_out in enumerate(layers):
            layer = dens(n, n_out, non_linear=non_linear)
            self.dense_layers.append(layer)
            setattr(self, f"dense{i}", layer)
            n = layer.shape_out
        # out layer
        i = len(self.dense_layers)
        n_out = len(self.classes)
        layer = dens(n, n_out, non_linear="identity")
        self.dense_layers.append(layer)
        setattr(self, f"dense{i}", layer)
        # migrate to appropriate device
        self.to(self.device)
        # reset optimization
        self.reset()

    @property
    def dump(self):
        parameters = dict()
        parameters["args"] = self.args
        parameters["x"] = self.x
        parameters["classes"] = self.classes
        parameters["mean"] = self.mean.tolist()
        parameters["std"] = self.std.tolist()
        dense = [d.dump for d in self.dense_layers]
        parameters["dense_layers"] = dense
        return {type(self).__name__: parameters, "version": _ml.__version__}

    @dump.setter
    def dump(self, other):
        if type(self).__name__ not in other.keys():
            raise ValueError(f"Expected dump from a '{type(self).__name__}'")
        other = other[type(self).__name__]
        self.x = other["x"]
        self.classes = other["classes"]
        self.mean = _np.array(other["mean"])
        self.std = _np.array(other["std"])
        self.args = other["args"]
        if self.args is not None:
            self._set_layers(*self.args)
        for i, d in enumerate(other["dense_layers"]):
            self.dense_layers[i].dump = d
        self.reset_optimizer()
