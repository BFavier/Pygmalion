import machine_learning.neural_networks.common as _common
import machine_learning._templates as _templates
import torch as _torch
import types as _types
import machine_learning as _ml
import machine_learning.neural_networks.layers as _layers
import numpy as _np


class Regressor(_common.NNtemplate, _templates.RegressorTemplate):
    """
    A model that predict scalar values from vectors of observations.

    Attributes:
    -----------
    x : list of str
        name of x columns
    mean_x : np.ndarray
        The mean value for normalization of input
    std_x : np.ndarray
        The standard deviation for normalization of input
    mean_y : np.ndarray
        The mean value for normalization of output
    std_y : np.ndarray
        The standard deviation for normalization of output
    dense_layers : list
        list of the dense layers of the model
    args : list or None
        list of arguments used to generate the model
    """

    def __init__(self):
        super().__init__()
        self.dense_layers = []
        self.x = []
        self.args = None

    def fit(self, x, y, layers=(32, 16, 8), validation=None,
            validation_fraction=0.2, restart=True, non_linear="relu",
            l_max=None, **kwargs):
        """
        Fit the model on the given data.

        Parameters:
        -----------
        x : pandas.DataFrame
            real valued observations
        y : iterable
            scalar predictions
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
        training = x, y
        # preprocess
        training, validation = self._validation_split(training, validation,
                                                      validation_fraction)
        res = self._preprocess(training, validation)
        self.mean_x, self.std_x, self.mean_y, self.std_y, self.x, _, _ = res
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
        self.mean_x, self.std_x, self.mean_y, self.std_y, self.x, \
            n_training, n_validation = res
        # create layers
        if restart:
            self._set_layers(layers, non_linear)
        # fitting
        self.train_loop(self._wrap(training), self._wrap(validation),
                        n_training, n_validation, **kwargs)

    def predict(self, X):
        """
        Make a prediction using the trained model.

        Parameters:
        -----------
        data : pandas.DataFrame

        Returns:
        --------
        Y : numpy.ndarray
        """
        X = X[self.x].values
        tensor = self._array_to_tensor(X, self.mean_x, self.std_x)
        with _torch.no_grad():
            output = self(tensor)
        Y = self._tensor_to_array(output, self.mean_y, self.std_y)
        return Y

    def forward(self, tensor):
        """Don't use this"""
        for i, dense in enumerate(self.dense_layers):
            tensor = dense(tensor)
        return tensor.view(-1)

    def _to_tensor(self, data):
        """Converts a dataframe into a tuple of tensors"""
        X, Y = data
        X, Y = X[self.x].values, _np.array(Y)
        X = self._array_to_tensor(X, mean=self.mean_x, std=self.std_x)
        Y = self._array_to_tensor(Y, mean=self.mean_y, std=self.std_y)
        return X, Y

    def _preprocess(self, training, validation):
        """preprocess the data"""
        # batchify if needed
        if not isinstance(training, _types.FunctionType):
            return self._preprocess(lambda: (yield training),
                                    lambda: (yield validation))
        # treat by batches
        sum_x, sum2_x, n_x = 0., 0., 0.
        sum_y, sum2_y, n_y = 0., 0., 0.
        x = None
        # treat training data
        for i, (X, Y) in enumerate(training()):
            if x is None:
                x = list(X.columns)
            for key in X.columns:
                if X[key].isna().any():
                    raise ValueError(f"At batch {i}: Found NaN in "
                                     f"training x['{key}']")
            if _np.isnan(Y).any():
                raise ValueError(f"At batch {i}: Found NaN in training y")
            sum_x, sum2_x, n_x = self._sums(X[x].values, sum_x, sum2_x, n_x)
            sum_y, sum2_y, n_y = self._sums(Y, sum_y, sum2_y, n_y)
        n_training = i+1
        # treat validation data
        for i, (X, Y) in enumerate(validation()):
            for key in X.columns:
                if X[key].isna().any():
                    raise ValueError(f"At batch {i}: Found NaN in "
                                     f"validation x['{key}']")
            if _np.isnan(Y).any():
                raise ValueError(f"At batch {i}: Found NaN in validation y")
            sum_x, sum2_x, n_x = self._sums(X.values, sum_x, sum2_x, n_x)
            sum_y, sum2_y, n_y = self._sums(Y, sum_y, sum2_y, n_y)
        n_validation = i+1
        # returns values
        mean_x, std_x = self._mean_std(sum_x, sum2_x, n_x)
        mean_y, std_y = self._mean_std(sum_y, sum2_y, n_y)
        return mean_x, std_x, mean_y, std_y, x, n_training, n_validation

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
        n_out = 1
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
        parameters["x"] = self.x
        parameters["mean_x"] = self.mean_x.tolist()
        parameters["std_x"] = self.std_x.tolist()
        parameters["mean_y"] = self.mean_y.tolist()
        parameters["std_y"] = self.std_y.tolist()
        parameters["args"] = self.args
        dense = [d.dump for d in self.dense_layers]
        parameters["dense_layers"] = dense
        return {type(self).__name__: parameters, "version": _ml.__version__}

    @dump.setter
    def dump(self, other):
        if type(self).__name__ not in other.keys():
            raise ValueError(f"Expected dump from a '{type(self).__name__}'")
        other = other[type(self).__name__]
        self.x = other["x"]
        self.mean_x = _np.array(other["mean_x"])
        self.std_x = _np.array(other["std_x"])
        self.mean_y = _np.array(other["mean_y"])
        self.std_y = _np.array(other["std_y"])
        self.args = other["args"]
        if self.args is not None:
            self._set_layers(*self.args)
        for i, d in enumerate(other["dense_layers"]):
            self.dense_layers[i].dump = d
        self.reset_optimizer()

    @property
    def cpp(self):
        ny = len(self.std_y)
        code = "#include <vector>\n"
        code += "#include <math.h>\n"
        code += "double relu(double x) {if (x > 0.) {return x;} return 0.;}\n"
        code += "double tanh(double x) {return std::tanh(x);}\n"
        code += "double identity(double x) {return x;}\n\n"
        out_type = "std::vector<double>" if ny > 1 else "double"
        args = ", ".join(["double "+x for x in self.x])
        code += f"{out_type} neural_network({args})\n"
        code += "{\n"
        # Normalization
        for i, x in enumerate(self.x):
            code += f"\t{x} = ({x}-{self.mean_x[i]})/{self.std_x[i]};\n"
        # Layers
        inputs = self.x
        for i, layer in enumerate(self.dense_layers):
            for j, weights in enumerate(layer.linear.weight):
                code += f"\tdouble nn_{i}_{j} = {layer.non_linear}("
                parts = [None]*len(weights)
                for k, var in enumerate(inputs):
                    w = float(weights[k])
                    parts[k] = f"{inputs[k]}*{w}"
                if layer.linear.bias is not None:
                    parts.append(f"{layer.linear.bias[j]}")
                code += " + ".join(parts) + ");\n"
            inputs = [f"nn_{i}_{j}" for j, _ in enumerate(layer.linear.weight)]
        # normalization
        for k in range(ny):
            code += f"\tnn_{i}_{k} = nn_{i}_{k}*{self.std_y[k]}" \
                    f" + {self.mean_y[k]};\n"
        # return value
        if ny == 1:
            code += f"\treturn nn_{i}_{0};\n"
        else:
            code += "return {"
            code += ", ".join([f"nn_{i}_{k}" for k in range(ny)])
            code += "};\n"
        code += "}"
        return code
