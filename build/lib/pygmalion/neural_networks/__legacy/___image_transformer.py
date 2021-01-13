import types as _types
import torch as _torch
import machine_learning as _ml
import machine_learning.neural_networks.common as _common
import machine_learning.neural_networks.layers as _layers
import numpy as _np


class ImageTransformer(_common.NNtemplate):
    """
    A model that applies a transformation over an image.

    Attributes:
    -----------
    convolution_layers : list
        The list of ConvolutionLayer objects
    channels_in : int
        channels of the input image
    channels_out : int
        channels of the output image
    mean_x : np.ndarray
        The mean value for normalization of each input channel
    std_x : np.ndarray
        The standard deviation for normalization of each input channel
    mean_y : np.ndarray
        The mean value for normalization of each output channel
    std_y : np.ndarray
        The standard deviation for normalization of each output channel
    args : list or None
        list of arguments used to generate the model
    """

    def __init__(self):
        super().__init__()
        self.convolution_layers = []
        self.channels_in = 0
        self.channels_out = 0
        self.args = None

    def fit(self, images, masks, validation=None, validation_fraction=0.2,
            windows=[], channels=[], restart=True, non_linear="relu",
            l_max=None, low_memory=False, **kwargs):
        """
        Fit the model on the given data.

        Parameters:
        -----------
        images : list of Image objects
            a list of machine_learning.Images objects
        masks : list of Image objects
            a list of arrays of shape (channels, height, width)
        validation : tuple of (images, masks) or None
            data used for validation
        validation_fraction : float
            if validation is None, training data is splited
        windows : list of tuple of int
            the convolution windows (height, width)
        channels : list of int
            channels number at output of each convolution
        restart : bool
            if False, the architecture is not modified
        non_linear : str or function
            non-linear function
        l_max : int or None
            maximum observations in a batch
        low_memory : bool
            If True, save memory at the cost of computation speed.
            Has effect only for deep networks.
        Returns:
        --------
        training, validation : tuples of (images, masks)

        Other parameters:
        -----------------
        **kwargs from self.train_loop(...)
        """
        # preprocess
        training = (images, masks)
        training, validation = self._validation_split(training, validation,
                                                      validation_fraction)
        res = self._preprocess(training, validation)
        self.mean_x, self.std_x, self.mean_y, self.std_y, self.channels_in, \
            self.channels_out, _, _ = res
        # create layers
        if restart:
            self._set_layers(windows, channels, non_linear, low_memory)
        # load data in memory
        train = self._to_tensor(training)
        val = self._to_tensor(validation)
        train, n_training = self._wrap_batch(train, l_max)
        val, n_validation = self._wrap_batch(val, l_max)
        # fitting
        self.train_loop(train, val, n_training, n_validation, **kwargs)
        # returning training and validation data
        return training, validation

    def fit_batch(self, training, validation, windows=[], channels=[],
                  restart=True, non_linear="relu", low_memory=False, **kwargs):
        """
        Fit the model on the given batches.

        Parameters:
        -----------
        training : generator function
            function that yields (images, labels)
        validation : generator function
            function that yields (images, labels)
        windows : list of tuple of int
            the convolution windows (height, width)
        channels : list of int
            channels number at output of each convolution
        restart : bool
            if False, the architecture is not modified
        non_linear : str or function
            non-linear function
        low_memory : bool
            If True, save memory at the cost of computation speed.
            Has effect only for deep networks.

        Other parameters:
        -----------------
        **kwargs from self.train_loop(...)
        """
        # preprocess
        res = self._preprocess(training, validation)
        self.mean_x, self.std_x, self.mean_y, self.std_y, self.channels_in, \
            self.channels_out, n_training, n_validation = res
        # create layers
        if restart:
            self._set_layers(windows, channels, non_linear, low_memory)
        # fitting
        self.train_loop(self._wrap(training), self._wrap(validation),
                        n_training, n_validation, **kwargs)

    def predict(self, images):
        """
        Make a prediction using the trained model.

        Parameters:
        -----------
        images : Image or list of Images

        Returns:
        --------
        labels : list
        """
        scalar = False
        if isinstance(images, _ml.Image):
            images = [images]
            scalar = True
        tensor = self._images_to_tensor(images, self.mean_x, self.std_x)
        with _torch.no_grad():
            output = self(tensor)
        masks = self._tensor_to_images(output, self.mean_y, self.std_y)
        if scalar:
            masks = masks[0]
        return masks

    def forward(self, tensor):
        """Don't use this"""
        for i, conv in enumerate(self.convolution_layers):
            tensor = conv(tensor)
        return tensor

    def _to_tensor(self, data):
        """Converts a tuple (images, masks) into a tuple of tensors"""
        images, masks = data
        X = self._images_to_tensor(images, mean=self.mean_x, std=self.std_x)
        Y = self._images_to_tensor(masks, mean=self.mean_y, std=self.std_y)
        return X, Y

    def _preprocess(self, training, validation):
        """preprocess the data (get image shape, normalization, ...)"""
        # batchify if needed
        if not isinstance(training, _types.FunctionType):
            return self._preprocess(lambda: (yield training),
                                    lambda: (yield validation))
        # treat by batches
        sum_x, sum2_x, n_x = 0., 0., 0.
        sum_y, sum2_y, n_y = 0., 0., 0.
        channels_in = set()
        channels_out = set()
        # treat training data
        for i, (images, masks) in enumerate(training()):
            sum_x, sum2_x, n_x = self._sums(images, sum_x, sum2_x, n_x)
            sum_y, sum2_y, n_y = self._sums(masks, sum_y, sum2_y, n_y)
            for image in images:
                channels_in = channels_in | {image.tensor.shape[0]}
            for mask in masks:
                channels_out = channels_out | {mask.tensor.shape[0]}
        n_training = i+1
        # treat validation data
        for i, (images, masks) in enumerate(validation()):
            sum_x, sum2_x, n_x = self._sums(images, sum_x, sum2_x, n_x)
            sum_y, sum2_y, n_y = self._sums(masks, sum_y, sum2_y, n_y)
            for image in images:
                channels_in = channels_in | {image.tensor.shape[0]}
            for mask in masks:
                channels_out = channels_out | {mask.tensor.shape[0]}
        n_validation = i+1
        # returns values
        if len(channels_in) > 1:
            raise ValueError("found different put channels count: "
                             f"{channels_in}")
        if len(channels_out) > 1:
            raise ValueError("found different output channels count: "
                             f"{channels_out}")
        mean_x, std_x = self._mean_std(sum_x, sum2_x, n_x)
        mean_y, std_y = self._mean_std(sum_y, sum2_y, n_y)
        channels_in, channels_out = channels_in.pop(), channels_out.pop()
        return (mean_x, std_x, mean_y, std_y, channels_in, channels_out,
                n_training, n_validation)

    def _set_layers(self, *args):
        """Remove the previous layers and create new ones"""
        windows, channels, non_linear, low_memory = args
        self.args = args
        # Deleting previous layers
        for i, _ in enumerate(self.convolution_layers):
            delattr(self, f"conv{i}")
        self.convolution_layers = []
        # Set convolution layers
        conv = _layers.ConvolutionLayer
        c_in = self.channels_in
        for i, c_out in enumerate(channels):
            window = windows[i]
            layer = conv(c_in, c_out, non_linear=non_linear,
                         convolution_window=window, padding_size="auto",
                         low_memory=low_memory,
                         stride=(1, 1), pooling_window=(1, 1))
            self.convolution_layers.append(layer)
            setattr(self, f"conv{i}", layer)
            c_in = c_out
        # out layer
        c_out = self.channels_out
        i = len(self.convolution_layers)
        layer = conv(c_in, c_out, non_linear="identity",
                     convolution_window=(1, 1), padding_size="auto",
                     stride=(1, 1), pooling_window=(1, 1))
        self.convolution_layers.append(layer)
        setattr(self, f"conv{i}", layer)
        # migrate to appropriate device
        self.to(self.device)
        # reset optimization
        self.reset()

    @property
    def dump(self):
        parameters = dict()
        parameters["channels_in"] = self.channels_in
        parameters["channels_out"] = self.channels_out
        parameters["args"] = self.args
        parameters["mean_x"] = self.mean_x.tolist()
        parameters["std_x"] = self.std_x.tolist()
        parameters["mean_y"] = self.mean_y.tolist()
        parameters["std_y"] = self.std_y.tolist()
        conv = [conv.dump for conv in self.convolution_layers]
        parameters["convolution_layers"] = conv
        return {type(self).__name__: parameters, "version": _ml.__version__}

    @dump.setter
    def dump(self, other):
        if type(self).__name__ not in other.keys():
            raise ValueError(f"Expected dump from a '{type(self).__name__}'")
        other = other[type(self).__name__]
        self.channels_in = other["channels_in"]
        self.channels_out = other["channels_out"]
        self.mean_x = _np.array(other["mean_x"])
        self.std_x = _np.array(other["std_x"])
        self.mean_y = _np.array(other["mean_y"])
        self.std_y = _np.array(other["std_y"])
        self.args = other["args"]
        if self.args is not None:
            self._set_layers(*self.args)
        for i, d in enumerate(other["convolution_layers"]):
            self.convolution_layers[i].dump = d
        self.reset_optimizer()
