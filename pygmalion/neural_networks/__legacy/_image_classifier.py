import machine_learning as _ml
import machine_learning._templates as _templates
import machine_learning.neural_networks.common as _common
import machine_learning.neural_networks.layers as _layers
import torch as _torch
import types as _types
import numpy as _np


class ImageClassifier(_common.NNtemplate, _templates.ClassifierTemplate):
    """
    A model that classify images.

    Attributes:
    -----------
    categories : list of categories
    convolution_layers : list of ConvolutionLayer
    dense_layers : list of DenseLayer
    input_shape : tuple of int
        shape of the input images formated as:
        (channels, height, width)
    args : list or None
        list of arguments used to generate the model
    """

    def __init__(self):
        super().__init__()
        self.convolution_layers = []
        self.dense_layers = []
        self.input_shape = (0, 0, 0)
        self.loss = _torch.nn.functional.cross_entropy
        self.args = None

    def fit(self, images, labels, validation=None, validation_fraction=0.2,
            windows=[], channels=[], strides=[], pooling=[], dense=[],
            l_max=None, restart=True, non_linear="relu", padded=False,
            **kwargs):
        """
        Fit the model on the given data.

        Parameters:
        -----------
        images : list of Image objects
            a list of machine_learning.Images objects
        labels : list
            a list of label object for each image
            (the labels can be a list of str for example)
        validation : tuple of (images, labels) or None
            data used for validation
        validation_fraction : float
            if validation is None, training data is splited
        windows : list of tuple of int
            the convolution windows (height, width)
        channels : list of int
            channels number at output of each convolution
        strides : list of tuple of int
            the convolution strides (y, x)
        pooling : list of tuple of int
            the polling windows (height, width)
        dense : list of int
            the dense layers
        l_max : int or None
            maximum observations in a batch
        restart : bool
            if False, the architecture is not modified
        non_linear : str or function
            non-linear function
        low_memory : bool
            If True, save memory at the cost of computation speed.
        padded : bool
            If True, a padding is applied before convolution.
            If padded is True, stride is (1, 1) and pooling is (1, 1),
            The tensor has the same shape at the output than at the input.

        Returns:
        --------
        training, validation : tuples of (images, labels)

        Other parameters:
        -----------------
        **kwargs from self.train_loop(...)
        """
        # preprocess
        training = (images, labels)
        training, validation = self._validation_split(training, validation,
                                                      validation_fraction)
        res = self._preprocess(training, validation)
        self.mean, self.std, self.input_shape, self.categories, _, _ = res
        # create layers
        if restart:
            self._set_layers(windows, channels, strides, pooling, dense,
                             non_linear, padded)
        # load data in memory
        train = self._to_tensor(training)
        val = self._to_tensor(validation)
        train, n_training = self._wrap_batch(train, l_max)
        val, n_validation = self._wrap_batch(val, l_max)
        # fitting
        self.train_loop(train, val, n_training, n_validation, **kwargs)
        # returning training and validation data
        return training, validation

    def fit_batch(self, training, validation, padded=False,
                  windows=[], channels=[], strides=[], pooling=[], dense=[],
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
        strides : list of tuple of int
            the convolution strides (y, x)
        pooling : list of tuple of int
            the polling windows (height, width)
        dense : list of int
            the dense layers
        restart : bool
            if False, the architecture is not modified
        non_linear : str or function
            non-linear function
        low_memory : bool
            If True, save memory at the cost of computation speed.
        padded : bool
            If True, a padding is applied before convolution.
            If padded is True, stride is (1, 1) and pooling is (1, 1),
            The tensor has the same shape at the output than at the input.

        Other parameters:
        -----------------
        **kwargs from self.train_loop(...)
        """
        # preprocess
        self.mean, self.std, self.input_shape, self.categories, n_training, \
            n_validation = self._preprocess(training, validation)
        # create layers
        if restart:
            self._set_layers(windows, channels, strides, pooling, dense,
                             non_linear, low_memory, padded)
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
        labels = self._tensor_to_labels(output)
        if scalar:
            labels = labels[0]
        return labels

    def deep_dream(self, layer, channel, iterations=100, save_frequency=10,
                   verbose=True, input_image=None, lr=1.0E2, L2=0., L1=0.):
        """
        Returns an image representation of what best activates
        the given channel of the given layer

        https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
        """
        # create image tensor
        if input_image is None:
            data = _np.random.randint(108, 148, size=self.input_shape)
            input_image = _ml.Image(tensor=data)
        # tensor = self._images_to_tensor(input_image, mean=self.mean,
        #                                 std=self.std)
        tensor = _np.expand_dims(input_image.tensor, axis=0)
        tensor = _torch.tensor(tensor, requires_grad=True,
                               dtype=_torch.float, device=self.device)
        mean = _torch.tensor(self.mean, dtype=_torch.float, device=self.device)
        std = _torch.tensor(self.std, dtype=_torch.float, device=self.device)
        # tensor = tensor.clone().detach().requires_grad_(True)
        tensor = _torch.nn.Parameter(tensor)
        print("max ", tensor.max().item(), "\tmin ", tensor.min().item())
        optimizer = _torch.optim.Adam([tensor], lr=lr, weight_decay=L2)
        # Set the hook on the model
        for module in self.convolution_layers:
            module.low_memory = False
        layers = self.convolution_layers+self.dense_layers
        watcher = _common.LayerActivationWatcher(layers[layer])
        # loop until the layer's activation is maximised
        images = [input_image]
        f = save_frequency
        if verbose:
            print()
        try:
            for i in range(iterations):
                optimizer.zero_grad()
                old = tensor.clone().detach()
                self((tensor.clamp(0., 255.) - mean)/std)
                loss = -(watcher.activation[:, channel]).mean()
                loss.backward()
                optimizer.step()
                print("max ", tensor.max().item(), "\tmin ",
                      tensor.min().item())
                # save the intermediate image
                if (i % f == 0) or i == iterations-1:
                    # images.append(self._tensor_to_images(tensor,
                    #               mean=self.mean, std=self.std)[0])
                    images.append(_ml.Image(tensor=tensor.detach().cpu()[0]))
                    if verbose:
                        print("image saved")
                # compute the change
                change = _torch.mean((tensor - old)**2).item()
                change /= _torch.mean(old**2).item()
                if verbose:
                    print(f"\riteration {i}: change={change:.3g}\t\t\t")
        except KeyboardInterrupt:
            pass
        if verbose:
            print()
        return images

    def forward(self, tensor):
        """Don't use this"""
        for i, conv in enumerate(self.convolution_layers):
            tensor = conv(tensor)
        tensor = tensor.view(len(tensor), -1)
        for i, dense in enumerate(self.dense_layers):
            tensor = dense(tensor)
        return tensor

    def _to_tensor(self, data):
        """Converts a tuple (images, labels) into a tuple of tensors"""
        images, labels = data
        X = self._images_to_tensor(images, mean=self.mean, std=self.std)
        Y = self._labels_to_tensor(labels)
        return X, Y

    def _preprocess(self, training, validation):
        """preprocess the data (get image shape, normalization, ...)"""
        # batchify if needed
        if not isinstance(training, _types.FunctionType):
            return self._preprocess(lambda: (yield training),
                                    lambda: (yield validation))
        # treat by batches
        sum_, sum2, n = 0., 0., 0.
        shapes = []
        categories = set()
        # treat training data
        for i, (images, labels) in enumerate(training()):
            sum_, sum2, n = self._sums(images, sum_, sum2, n)
            categories = categories | set(labels)
            for image in images:
                shape = image.tensor.shape
                if shape not in shapes:
                    shapes.append(shape)
        n_training = i+1
        # treat validation data
        for i, (images, labels) in enumerate(validation()):
            sum_, sum2, n = self._sums(images, sum_, sum2, n)
            categories = categories | set(labels)
            for image in images:
                shape = image.tensor.shape
                if shape not in shapes:
                    shapes.append(shape)
        n_validation = i+1
        mean, std = self._mean_std(sum_, sum2, n)
        if len(shapes) > 1:
            raise ValueError(f"Expected a single image shape, found {shapes}")
        return mean, std, shapes.pop(), list(categories), \
            n_training, n_validation

    def _set_layers(self, *args):
        """Remove the previous layers and create new ones"""
        windows, channels, strides, pooling, dense, non_linear, \
            padded = args
        self.args = args
        # Deleting previous layers
        for i, _ in enumerate(self.convolution_layers):
            delattr(self, f"conv{i}")
        self.convolution_layers = []
        for i, _ in enumerate(self.dense_layers):
            delattr(self, f"dense{i}")
        self.dense_layers = []
        # Set convolution layers
        conv = _layers.ConvolutionLayer
        c, h, w = self.input_shape
        for i, c_out in enumerate(channels):
            layer = conv(c, c_out, non_linear=non_linear,
                         convolution_window=windows[i],
                         padding_size="auto" if padded else (0, 0, 0, 0),
                         stride=strides[i], pooling_window=pooling[i])
            self.convolution_layers.append(layer)
            setattr(self, f"conv{i}", layer)
            c, h, w = layer.shape_out((c, h, w))
            if c*h*w == 0:
                raise ValueError(f"image shape is {(c, h, w)} after conv{i}")
        # Set dense layers
        dens = _layers.DenseLayer
        n = c*h*w
        for i, n_out in enumerate(dense):
            layer = dens(n, n_out, non_linear=non_linear)
            self.dense_layers.append(layer)
            setattr(self, f"dense{i}", layer)
            n = layer.shape_out
        # out layer
        i = len(self.dense_layers)
        n_out = len(self.categories)
        layer = dens(n, n_out, non_linear="identity")
        self.dense_layers.append(layer)
        setattr(self, f"dense{i}", layer)
        # migrate to appropriate device
        self.to(self.device)
        # reset optimization
        self.reset()

    @property
    def shapes(self):
        """return the shape of the image after the successive layers"""
        shape = self.input_shape
        shapes = [shape]
        for conv in self.convolution_layers:
            shape = conv.shape_out(shape)
            shapes.append(shape)
        for dense in self.dense_layers:
            shapes.append(dense.linear.weight.shape[0])
        return shapes

    @property
    def dump(self):
        parameters = dict()
        parameters["args"] = self.args
        parameters["categories"] = self.categories
        parameters["mean"] = self.mean.tolist()
        parameters["std"] = self.std.tolist()
        parameters["input_shape"] = self.input_shape
        conv = [c.dump for c in self.convolution_layers]
        parameters["convolution_layers"] = conv
        dense = [d.dump for d in self.dense_layers]
        parameters["dense_layers"] = dense
        return {type(self).__name__: parameters, "version": _ml.__version__}

    @dump.setter
    def dump(self, other):
        if type(self).__name__ not in other.keys():
            raise ValueError(f"Expected dump from a '{type(self).__name__}'")
        other = other[type(self).__name__]
        self.categories = other["categories"]
        self.mean = _np.array(other["mean"])
        self.std = _np.array(other["std"])
        self.input_shape = other["input_shape"]
        self.args = other["args"]
        if self.args is not None:
            self._set_layers(*self.args)
        for i, d in enumerate(other["convolution_layers"]):
            self.convolution_layers[i].dump = d
        for i, d in enumerate(other["dense_layers"]):
            self.dense_layers[i].dump = d
        self.reset_optimizer()
