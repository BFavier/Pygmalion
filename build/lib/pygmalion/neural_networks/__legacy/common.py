import types as _types
import math as _math
import torch as _torch
import numpy as _np
import matplotlib.pyplot as _plt
import machine_learning as _ml
import machine_learning.neural_networks.layers as _layers
import machine_learning._utilities as _utils


functions = {'relu': _torch.nn.functional.relu,
             'tanh': _torch.tanh,
             'sigmoid': _torch.sigmoid,
             'hard tanh': _torch.nn.functional.hardtanh,
             'log sigmoid': _torch.nn.functional.logsigmoid,
             'identity': lambda x: x}


def as_function(non_linear):
    """
    Returns a function corresponding to the given name.
    If non_linear is already a function, returns it.
    """
    if isinstance(non_linear, _types.FunctionType):
        return non_linear
    if non_linear not in functions.keys():
        raise KeyError(f"got {non_linear} for 'non_linear'"
                       f", expected one of {tuple(functions.keys())}")
    return functions[non_linear]


def as_function_name(non_linear):
    """returns the name corresponding to the function"""
    if not isinstance(non_linear, _types.FunctionType):
        raise TypeError(f"Expected a function but got '{type(non_linear)}'")
    for name, func in functions.items():
        if func == non_linear:
            return name
    else:
        return "unknown"


class NNtemplate(_torch.nn.Module):
    """
    A template of neural network model

    Attributes:
    -----------
    GPU : bool
        Whether to use GPU or not
    history : dict
        dictionary with history of fitting
    loss : function
        the loss(output, target) function
    optimizer : torch.optim.Optimizer
        the optimizer
    L2 : float
        L2 normalization factor
    learning_rate : float
        The learning rate of the optimization
    fitted : bool
        Whether the model was fitted already
    """

    def __init__(self):
        """initialize default attributes"""
        super().__init__()
        self.GPU = _torch.cuda.is_available()
        self.history = self._empty_history()
        self.loss = _torch.nn.functional.mse_loss
        self._L2 = 0.
        self._L1 = 0.
        self._learning_rate = 1.0E-3
        dummy = [_torch.nn.Parameter(_torch.tensor([1.]))]
        self.optimizer = _torch.optim.Adam(dummy, weight_decay=self.L2,
                                           lr=self.learning_rate)
        self.fitted = False

    def train_loop(self, training, validation, n_training, n_validation,
                   n_epochs=100, learning_rate=1.0E-3, L1=0., L2=0.,
                   keep_best=True, patience=None, verbose=True):
        """
        train using batch gradient descent on provided mini-batchs

        Parameters:
        -----------
        training : generator function
            a function that yield (input, target)
        validation : generator function
            a function that yields (input, target)
        n_training : int
            number of batchs yielded by 'training'
        n_validation : int
            number of batchs yielded by 'validation'

        Other parameters:
        -----------------
        n_epochs : int
            number of epoch to perform on training data
        learning_rate : float
            Learning rate of the model
        keep_best : bool
            If True, keep the epoch of best validation loss
        patience : int
            Stops training if validation loss has not decreased in
            'patience' epochs
        GPU : bool
            If True, trains on GPU
        verbose : bool
            If True, prints during fitting
        """
        # activate training mode
        self.train()
        # If a tuple of itterables is passed, transform them into a generator
        if not isinstance(training, _types.FunctionType):
            return self.train_loop(lambda: (yield training),
                                   lambda: (yield validation),
                                   1, 1, n_epochs=n_epochs,
                                   learning_rate=learning_rate, L1=L1, L2=L2,
                                   keep_best=keep_best, patience=patience,
                                   verbose=verbose)
        # setting hyper-parameters
        _torch.backends.cudnn.benchmark = True
        self.learning_rate = learning_rate
        self.L2 = L2
        self.L1 = L1
        # loading data
        best_params = self.state_dict()
        optim_state = self.optimizer.state_dict()
        training_loss, validation_loss = [], []
        best_epoch = self.history["best epoch"]
        if best_epoch is not None:
            best_loss = self.history["validation loss"][best_epoch]
        else:
            best_loss = float("inf")
        # Stoping training if user press ctrl+c
        try:
            # Looping on epochs
            k = 0 if best_epoch is None else best_epoch+1
            for epoch in range(k, k+n_epochs):
                if verbose:
                    print(f"Epoch {epoch} : ", end="")
                # looping on training mini-batchs
                mean_loss = []
                self.optimizer.zero_grad()
                for batch, (input, target) in enumerate(training()):
                    output = self(input)
                    loss = self.loss(output, target)
                    (loss/n_training + self._L1_reg()).backward()
                    mean_loss.append(loss.item())
                    # printing progress
                    if verbose:
                        self._print_progress(batch, n_training)
                self.optimizer.step()
                training_loss.append(_np.mean(mean_loss))
                if batch+1 != n_training:
                    raise ValueError(f"training yielded {batch+1} batchs, "
                                     f"but {n_training} were expected")
                # looping on validation mini-batchs
                mean_loss = []
                for batch, (input, target) in enumerate(validation()):
                    with _torch.no_grad():
                        output = self(input)
                        loss = self.loss(output, target)
                        mean_loss.append(loss.item())
                validation_loss.append(_np.mean(mean_loss))
                # printing progress
                if verbose:
                    print(f" training={training_loss[-1]:.3g}"
                          f" validation={validation_loss[-1]:.3g}")
                # Saving state
                if (validation_loss[-1] < best_loss) or not keep_best:
                    best_epoch = epoch
                    best_loss = validation_loss[-1]
                    best_params = self.state_dict()
                    optim_state = self.optimizer.state_dict()
                # Interupt training if no improvement
                if not(patience is None) and (epoch - best_epoch > patience):
                    break
        except KeyboardInterrupt:
            pass
        if verbose:
            print()
        # Returning results
        self.fitted = True
        self.history["training loss"].extend(training_loss)
        self.history["validation loss"].extend(validation_loss)
        self.history["best epoch"] = best_epoch if keep_best else epoch
        # Loading saved state
        if keep_best:
            self.load_state_dict(best_params)
            self.optimizer.load_state_dict(optim_state)
        # activate evaluation mode
        self.eval()

    def plot_history(self, ax=None, log=False):
        """
        Plot the history of the fitting
        """
        if ax is None:
            _plt.figure()
            ax = _plt.subplot(1, 1, 1)
        N = min(len(self.history["training loss"]),
                len(self.history["validation loss"]))
        epochs = [i for i in range(0, N)]
        if not self.history["best epoch"] is None:
            ax.axvline(x=self.history["best epoch"], color="k")
        ax.plot(epochs, self.history["training loss"][0:N], linestyle="",
                marker=".", color="C0",
                label="training loss (batchs average)")
        ax.plot(epochs, self.history["validation loss"][0:N], linestyle="",
                marker=".", color="C1",
                label="validation loss (batchs average)")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.get_figure().tight_layout()
        return ax

    def seed(self, seed):
        """
        Initialize the seed for the random numbers generation.
        This calls numpy.random.seed and torch.manuel_seed

        Kwargs:
            seed : int or None
                The seed to use
        """
        if seed is not None:
            _np.random.seed(seed)
            _torch.manual_seed(seed)
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False

    def reset(self):
        """
        Reset the model to an untrained state.

        Calls reset optimizer under the hood.
        """
        self.history = self._empty_history()
        self.fitted = False
        # reset parameters
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        # reset optimizer
        self.reset_optimizer()

    def reset_optimizer(self):
        """
        Reset the optimizer.

        If some parameters are added to or deleted from the model,
        this must be called. Otherwise optimizer won't aknowledge it.
        """
        opt_class = type(self.optimizer)
        if next(self.parameters(), None) is None:
            parameters = [_torch.nn.Parameter(_torch.tensor([1.]))]
        else:
            parameters = self.parameters()
        self.optimizer = opt_class(parameters,
                                   weight_decay=self.L2,
                                   lr=self.learning_rate)

    def forward(self, input):
        """Implement me"""
        raise Exception("Not implemented")

    def _print_progress(self, batch, n_training):
        """print progress during 'train_loop_batch' """
        now = batch/n_training
        after = (batch+1)/n_training
        k_now = _math.floor(10*now)
        k_after = _math.floor(10*after)
        for i in range(k_now, k_after):
            print(u"\u2588", end="")

    def _empty_history(self):
        """default value for the fitting history"""
        return {"training loss": [],
                "validation loss": [],
                "best epoch": None}

    def _dataframe_to_arrays(self, df, x, y, validation=None):
        """converts dataframe data to training and validation"""
        training = df[x].values, df[y].values
        if df is not None:
            validation = validation[x].values, validation[y].values
        return training, validation

    def _validation_split(self, training, validation, validation_fraction=0.2):
        """
        if validation is None, split training data
        into training and validation

        Parameters:
        -----------
        training : tuple of iterables
            tuple of X, Y for training
        validaton : tuple of iterables, or None
            tuple of X, Y for validation
        validation_fraction : float
            fraction of validation data if validation is None

        Returns:
        --------
            training, validation : tuple of iterables
        """
        if validation is None:
            X, Y = training
            assert len(X) == len(Y), "X and Y must have the same dimensions to perform validation split"
            n_total = len(X)
            index = _np.random.permutation(n_total)
            n_validation = int(validation_fraction*n_total)
            if n_validation == 0:
                raise ValueError("Validation split led to 0 validation data")
            if n_validation == n_total:
                raise ValueError("Validation split led to 0 training data")
            X_val = _utils.subset(X, index[:n_validation])
            Y_val = _utils.subset(Y, index[:n_validation])
            X_train = _utils.subset(X, index[n_validation:])
            Y_train = _utils.subset(Y, index[n_validation:])
            training = X_train, Y_train
            validation = X_val, Y_val
        return training, validation

    def _sums(self, X, sum_=0., sum2=0., n=0.):
        """calculate sum and sum of squares of X. Used for normalization"""
        if (isinstance(X, list) or
           isinstance(X, tuple)) and isinstance(X[0], _ml.Image):
            X = _np.array([image.tensor for image in X])
        else:
            X = _np.array(X)
        n += _np.product([k for i, k in enumerate(X.shape) if i != 1])
        axis = tuple([i for i in range(0, len(X.shape)) if i != 1])
        shape = tuple([-1]+[1 for _ in X.shape[2:]])
        sum_ += _np.sum(X, axis=axis).reshape(*shape)
        sum2 += _np.sum(X**2, axis=axis).reshape(*shape)
        return sum_, sum2, n

    def _mean_std(self, sum_, sum2, n):
        """calculate mean and std from sums"""
        mean = sum_ / n
        std = _np.sqrt(sum2 / n - mean**2)
        return mean, std

    def _images_to_tensor(self, images, mean=0., std=1.):
        """converts an image to a normalized tensor"""
        if isinstance(images, _ml.Image):
            images = [images]
        data = [(image.tensor-mean)/std for image in images]
        return _torch.tensor(data, dtype=_torch.float, device=self.device)

    def _tensor_to_images(self, tensor, mean=0., std=1.):
        """converts a normalized tensor to an image"""
        tensor = tensor.cpu().detach().numpy()
        tensor = tensor*std + mean
        images = [_ml.Image(tensor=data) for data in tensor]
        return images

    def _array_to_tensor(self, arrays, mean=0., std=1.):
        """converts an array to a normalized tensor"""
        data = (arrays-mean)/std
        return _torch.tensor(data, dtype=_torch.float, device=self.device)

    def _tensor_to_array(self, tensor, mean=0., std=1.):
        """converts a normalized tensor to array"""
        array = tensor.cpu().detach().numpy()
        array = array*std + mean
        return array

    def _labels_to_tensor(self, labels):
        Y = [self.categories.index(label) for label in labels]
        return _torch.tensor(Y, dtype=_torch.long, device=self.device)

    def _tensor_to_labels(self, tensor):
        indexes = _torch.argmax(tensor, dim=1)
        labels = [self.categories[index] for index in indexes]
        return labels

    def _segmented_images_to_tensor(self, segmented):
        """convert grayscale segmented images to a tensor"""
        S = [image.data for image in segmented]
        Y = [_np.argmax([_np.all(s == c, axis=-1)
             for c in self.categories], axis=0)
             for s in S]
        return _torch.tensor(Y, dtype=_torch.long, device=self.device)

    def _tensor_to_segmented_images(self, tensor):
        data = tensor.detach().cpu().numpy()
        data = _np.argmax(data, axis=1)
        data = _np.array(self.categories)[data]
        return [_ml.Image(data=d) for d in data]

    def _to_tensor(self, data):
        """converts a tuple (X, Y) into a tuple of tensors"""
        raise NotImplementedError("'_to_tensor' is called but not implemented")

    def _wrap(self, generator):
        """return a generator of tuple of tensors from a generator of (X, Y)"""
        def wrapper():
            for data in generator():
                yield self._to_tensor(data)
        return wrapper

    def _wrap_batch(self, data, n_max=None):
        """
        wrap a tuple of tensors into a generator yielding batchs
        of at most n_max observations

        parameters:
        -----------
        data : tuple of tensors
            the training or validation data
        n_max : int or None
            the maximum number of observation in a single batch
        """
        X, Y = data
        if n_max is not None:
            n_batchs = _math.ceil(len(X)/n_max)
        else:
            n_batchs = 1

        def wrapper():
            step = len(X)/n_batchs
            stop = 0
            for i in range(1, n_batchs+1):
                start = stop
                stop = int(i*step)
                yield X[start:stop], Y[start:stop]
        return wrapper, n_batchs

    def _L1_reg(self):
        """Returns the L1 regularization term of the neural network
        (L1 norm of all parameters times the factor)"""
        if self.L1 == 0.:
            return 0.
        l1_n = [(p.abs().sum(), _np.product(p.shape))
                for p in self.parameters()]
        l1, n = zip(*l1_n)
        mean_l1_norm = sum(l1)/sum(n)
        return self.L1*mean_l1_norm

    @property
    def GPU(self):
        return self._GPU

    @GPU.setter
    def GPU(self, other):
        self._GPU = other
        assert not(self.GPU) or _torch.cuda.is_available(), \
            "CUDA is not available on this machine. Unable to run on GPU."
        self.to(self.device)

    @property
    def device(self):
        """returns the device the model should train on"""
        if self.GPU:
            return _torch.device("cuda:0")
        else:
            return _torch.device("cpu")

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, other):
        self._learning_rate = other
        for group in self.optimizer.param_groups:
            group["lr"] = other

    @property
    def L2(self):
        return self._L2

    @L2.setter
    def L2(self, other):
        self._L2 = other
        for group in self.optimizer.param_groups:
            group["weight_decay"] = other

    @property
    def VRAM(self):
        """current VRAM usage (allocated and cached in capability %)"""
        t = _torch.cuda.get_device_properties(self.device).total_memory
        a = _torch.cuda.memory_allocated(self.device)
        allocated = a/t * 100.
        return allocated

    @property
    def VRAM_peak(self):
        """peak allocation % of VRAM since last call"""
        t = _torch.cuda.get_device_properties(self.device).total_memory
        a = _torch.cuda.max_memory_allocated(self.device)
        _torch.cuda.reset_max_memory_allocated(self.device)
        allocated = a/t * 100.
        return allocated


class LayerActivationWatcher():
    """
    visualize the intermediate results of a neural network.
    For deep dreaming applications.

    Parameters:
    -----------
    activation : torch.tensor or None
        activation of the target layer at last call.
    """
    def __init__(self, layer):
        self.activation = None
        if isinstance(layer, _layers.ConvolutionLayer):
            module = layer.convolution
        elif isinstance(layer, _layers.DenseLayer):
            module = layer.linear
        else:
            raise NotImplementedError("DeepDream is not implemented for "
                                      f"layer '{type(layer)}'")
        self.hook = module.register_forward_hook(self.hook_function)

    def __del__(self):
        self.hook.remove()

    def hook_function(self, module, input, output):
        self.activation = output
