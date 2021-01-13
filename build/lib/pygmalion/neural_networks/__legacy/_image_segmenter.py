
import types as _types
import torch as _torch
import machine_learning as _ml
import machine_learning._templates as _templates
import machine_learning.neural_networks.common as _common
import machine_learning.neural_networks.layers as _layers
import numpy as _np


class ImageSegmenter(_common.NNtemplate, _templates.ModelTemplate):
    """
    A model that returns a segmented grayscale image.
    Each gray level of the output corresponds to a class.
    (for example output can be a binarized 0 and 255 gray levels image)

    Attributes:
    -----------
    categories : list of uint8
        gray levels the output image can be segmented into
    convolution_layers : list
        The list of ConvolutionLayer objects
    channels_in : int
        channels of the input image
    mean : np.ndarray
        The mean value for normalization of each input channel
    std : np.ndarray
        The standard deviation for normalization of each input channel
    args : list or None
        list of arguments used to generate the model
    """

    def __init__(self):
        super().__init__()
        self.convolution_layers = []
        self.upsampling_layers = []
        self.out_layer = None
        self.channels_in = 0
        self.loss = _torch.nn.functional.cross_entropy
        self.args = None

    def fit(self, images, masks, validation=None, validation_fraction=0.2,
            down_windows=[], down_channels=[], pooling_windows=[],
            up_windows=[], up_channels=[], restart=True, non_linear="relu",
            l_max=None, **kwargs):
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
        (self.mean, self.std, self.channels_in, self.categories,
            frequ, _, _) = res
        # redefine loss with correct weights
        weights = _torch.Tensor([1/f for f in frequ])
        self.loss = _torch.nn.CrossEntropyLoss(weight=weights)
        # create layers
        if restart:
            self._set_layers(down_windows, down_channels, pooling_windows,
                             up_windows, up_channels, non_linear)
        # load data in memory
        train = self._to_tensor(training)
        val = self._to_tensor(validation)
        train, n_training = self._wrap_batch(train, l_max)
        val, n_validation = self._wrap_batch(val, l_max)
        # fitting
        self.train_loop(train, val, n_training, n_validation, **kwargs)
        # returning training and validation data
        return training, validation

    def fit_batch(self, training, validation, down_windows=[],
                  down_channels=[], pooling_windows=[], up_windows=[],
                  up_channels=[], restart=True, non_linear="relu", **kwargs):
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
        (self.mean, self.std, self.channels_in, self.categories,
            frequ, n_training, n_validation) = res
        # redefine loss with correct weights
        weights = _torch.Tensor([1/f for f in frequ])
        self.loss = _torch.nn.CrossEntropyLoss(weight=weights)
        # create layers
        if restart:
            self._set_layers(down_windows, down_channels, pooling_windows,
                             up_windows, up_channels, non_linear)
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
        tensor = self._images_to_tensor(images, self.mean, self.std)
        with _torch.no_grad():
            output = self(tensor)
        masks = self._tensor_to_segmented_images(output)
        if scalar:
            masks = masks[0]
        return masks

    def forward(self, tensor):
        """Don't use this"""
        downsampling = []
        for conv in self.convolution_layers:
            downsampling.append(tensor)
            tensor = conv(tensor)
        downsampling.reverse()
        for i, deconv in enumerate(self.upsampling_layers):
            tensor = deconv(tensor, downsampling[i])
        return self.out_layer(tensor)

    def _to_tensor(self, data):
        """Converts a tuple (images, masks) into a tuple of tensors"""
        images, masks = data
        X = self._images_to_tensor(images, mean=self.mean, std=self.std)
        Y = self._segmented_images_to_tensor(masks)
        return X, Y

    def _preprocess(self, training, validation):
        """preprocess the data (get image shape, normalization, ...)"""
        # batchify if needed
        if not isinstance(training, _types.FunctionType):
            return self._preprocess(lambda: (yield training),
                                    lambda: (yield validation))
        # treat by batches
        sum_, sum2, n = 0., 0., 0.
        channels_in = set()
        categories = []
        frequencies = []
        # treat training data
        for i, (images, masks) in enumerate(training()):
            sum_, sum2, n = self._sums(images, sum_, sum2, n)
            for image in images:
                channels_in = channels_in | {image.tensor.shape[0]}
            masks = _np.concatenate([mask.data for mask in masks])
            masks = masks.reshape((-1,) + masks.shape[2:])
            uniques, counts = _np.unique(masks, axis=0, return_counts=True)
            for u, c in zip(uniques, counts):
                u = u.tolist()
                try:
                    index = categories.index(u)
                except ValueError:
                    index = len(categories)
                    categories.append(u)
                    frequencies.append(c)
                finally:
                    frequencies[index] += c
        n_training = i+1
        # treat validation data
        for i, (images, masks) in enumerate(validation()):
            sum_, sum2, n = self._sums(images, sum_, sum2, n)
            for image in images:
                channels_in = channels_in | {image.tensor.shape[0]}
            masks = _np.concatenate([mask.data for mask in masks])
            masks = masks.reshape((-1,) + masks.shape[2:])
            uniques, counts = _np.unique(masks, axis=0, return_counts=True)
            for u, c in zip(uniques, counts):
                u = u.tolist()
                try:
                    index = categories.index(u)
                except ValueError:
                    index = len(categories)
                    categories.append(u)
                    frequencies.append(c)
                finally:
                    frequencies[index] += c
        n_validation = i+1
        # treating categories
        s = sum(frequencies)
        frequencies = [f/s for f in frequencies]
        # returns values
        if len(channels_in) > 1:
            raise ValueError("found different put channels count: "
                             f"{channels_in}")
        mean, std = self._mean_std(sum_, sum2, n)
        return (mean, std, channels_in.pop(), categories, frequencies,
                n_training, n_validation)

    def _set_layers(self, *args):
        """Remove the previous layers and create new ones"""
        (down_windows, down_channels, pooling_windows, up_windows, up_channels,
            non_linear) = args
        self.args = args
        # Deleting previous layers
        for i, _ in enumerate(self.convolution_layers):
            delattr(self, f"conv{i}")
        self.convolution_layers = []
        for i, _ in enumerate(self.upsampling_layers):
            delattr(self, f"upsample{i}")
        self.upsampling_layers = []
        # Setting alias
        conv = _layers.ConvolutionLayer
        up = _layers.UpsamplingLayer
        # Set convolution layers
        c_in = self.channels_in
        for i, c_out in enumerate(down_channels):
            window = down_windows[i]
            pooling = pooling_windows[i]
            layer = conv(c_in, c_out, non_linear=non_linear,
                         convolution_window=window, padding_size="auto",
                         stride=(1, 1), pooling_window=pooling)
            self.convolution_layers.append(layer)
            setattr(self, f"conv{i}", layer)
            c_in = c_out
        # Set upsampling_layer
        down_channels.reverse()
        down_channels = down_channels[1:] + [self.channels_in]
        for i, c_out in enumerate(up_channels):
            window = up_windows[i]
            layer = up(c_in+down_channels[i], c_out, non_linear=non_linear,
                       convolution_window=window)
            self.upsampling_layers.append(layer)
            setattr(self, f"upsample{len(self.upsampling_layers)}", layer)
            c_in = c_out
        # out layer
        c_out = len(self.categories)
        layer = conv(c_in, c_out, non_linear="identity",
                     convolution_window=(1, 1), padding_size="auto",
                     stride=(1, 1), pooling_window=(1, 1))
        self.out_layer = layer
        # migrate to appropriate device
        self.to(self.device)
        # reset optimization
        self.reset()

    @property
    def dump(self):
        parameters = dict()
        parameters["categories"] = self.categories
        parameters["mean"] = self.mean.tolist()
        parameters["std"] = self.std.tolist()
        parameters["args"] = self.args
        parameters["channels_in"] = self.channels_in
        conv = [conv.dump for conv in self.convolution_layers]
        parameters["convolution_layers"] = conv
        up = [up.dump for up in self.upsampling_layers]
        parameters["upsampling_layers"] = up
        parameters["out_layer"] = self.out_layer.dump
        return {type(self).__name__: parameters, "version": _ml.__version__}

    @dump.setter
    def dump(self, other):
        if type(self).__name__ not in other.keys():
            raise ValueError(f"Expected dump from a '{type(self).__name__}'")
        other = other[type(self).__name__]
        self.categories = other["categories"]
        self.mean = _np.array(other["mean"])
        self.std = _np.array(other["std"])
        self.args = other["args"]
        self.channels_in = other["channels_in"]
        if self.args is not None:
            self._set_layers(*self.args)
        for i, d in enumerate(other["convolution_layers"]):
            self.convolution_layers[i].dump = d
        for i, d in enumerate(other["upsampling_layers"]):
            self.upsampling_layers[i].dump = d
        self.out_layer.dump = other["out_layer"]
        self.reset_optimizer()
