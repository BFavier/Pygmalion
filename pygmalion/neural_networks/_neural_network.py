import math
import torch
import warnings
import matplotlib.pyplot as plt
import torch.nn.parallel as parallel
from typing import Union, Tuple, List, Optional, Callable
from .._model import Model


class LossModule(torch.nn.Module):
    """
    A wrapper around the module of the model that evaluates the loss
    in the forward pass

    This is required for balanced memory use with multi GPU training
    """

    def __init__(self, model: 'NeuralNetwork'):
        super().__init__()
        self.module = model.module

    def forward(self, x, y_target, weights=None):
        y_pred = self.module(x)
        return self.module.loss(y_pred, y_target, weights=weights).unsqueeze(0)


class NeuralNetwork(Model):
    """
    Attributes
    ----------
    module : torch.nn.Module
        the underlying torch module of the model
    optimizer : torch.optim
        the optimizer used for training
    """
    ModuleType: type = None

    @classmethod
    def from_dump(cls, dump: dict) -> 'NeuralNetwork':
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        obj.module = cls.ModuleType.from_dump(dump["module"])
        obj.history = dump["history"]
        obj.optimization_method = dump["optimization method"]
        obj.GPU = dump["GPU"]
        return obj

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __init__(self, *args,
                 GPU: Union[None, int, List[int]] = None,
                 optimization_method: str = "Adam",
                 **kwargs):
        """
        NeuralNetwork parameters
        ------------------------
        GPU : None or int or list of int
            The indice of the GPU to evaluate the model on
            Set None to evaluate on cpu
        optimization_method : str
            The name of the optimization method ("Adam", "SGD", ...)
        """
        self.module = self.ModuleType(*args, **kwargs)
        self.GPU = GPU
        self.optimization_method = optimization_method
        self.history = {"training loss": [],
                        "validation loss": [],
                        "epochs": [],
                        "best epoch": None}

    def __call__(self, X, batch_size: Optional[int] = None):
        """
        Returns the model's evaluation on the given input

        Parameters:
        -----------
        X : Any
            The input X of the model.
            it's type depend on the neural network type.
        batche_size : int or None
            Parameter used to save on memory
            The maximum number of observations to evaluate at once
            Or None to process everything in one go

        Returns
        -------
        Any :
            The returned Y value of the model.
            it's type depends on the neural network type.
        """
        torch.cuda.empty_cache()
        self.module.eval()
        data, _ = self._batch_generator((X, None), None, batch_size, None,
                                        self.device, shuffle=False)
        with torch.no_grad():
            y = torch.cat([self.module(batch) for batch, _, _ in data()],
                          dim=0)
        return self._tensor_to_y(y)

    def train(self, training_data: tuple,
              validation_data: Union[tuple, None] = None,
              n_epochs: int = 1000,
              learning_rate: float = 1.0E-3,
              patience: Optional[int] = None,
              batch_size: Optional[int] = None,
              n_batches: Optional[int] = None,
              L1: Optional[float] = None,
              L2: Optional[float] = None,
              keep_best: bool = True,
              verbose: bool = True):
        """
        Trains a neural network model.

        Parameters
        ----------
        training_data : tuple or callable
            The data used to fit the model on.
            A tuple of (x, y[, weights]) or a callable that yields them.
            The type of each element depends on the model kind.
        validation_data : tuple or callable or None
            The data used to test for early stoping.
            Similar to training_data or None
        n_epochs : int
            The maximum number of epochs
        learning_rate : float
            The learning rate used to update the parameters
        patience : int or None
            The number of epochs before early stopping
            (if no improvement for 'patience' epochs, stops training early)
            If None, no early stoping is performed
        batch_size : int or None
            Maximum size of the batchs
            Or None to process the full data in one go
        n_batches : int or None
            If not None, set the number of batches processed per epoch
            Otherwise the whole data is processed
        L1 : float or None
            L1 regularization added to the loss function
        L2 : float or None
            L2 regularization added to the loss function
        keep_best : bool
            If True, the model is checkpointed at each epoch if there was
            improvement,
            and the best model is loaded back at the end of training
        verbose : bool
            If True the loss are displayed at each epoch
        """
        self._set_learning_rate(learning_rate)
        self.module.train()
        # Converts training/validation data to generators
        if batch_size is None and n_batches is None:
            device = self.device
        else:
            device = torch.device("cpu")
        training_data, validation_data = self._batch_generator(
            training_data, validation_data, batch_size, n_batches, device)
        # Wrap the module if training on multi GPU
        if isinstance(self.GPU, list):
            loss_module = parallel.DataParallel(LossModule(self),
                                                device_ids=self.GPU)
        else:
            loss_module = LossModule(self)
        # Initializing
        if self.history["best epoch"] is None:
            best_loss = float("inf")
            best_epoch = 0
        else:
            best_epoch = self.history["best epoch"]
            i = self.history["epochs"].index(best_epoch)
            best_loss = self.history["validation loss"][i]
            if best_loss is None:
                best_loss = float("inf")
            for v in ["validation loss", "training loss", "epochs"]:
                self.history[v] = self.history[v][:i+1]
        best_state = self._get_state()
        # trains the model, stops if the user press 'ctrl+c'
        self._training_loop(loss_module, training_data, validation_data,
                            n_epochs, patience, verbose, keep_best, L1, L2,
                            best_epoch, best_loss, best_state)

    def plot_history(self, ax=None, log: bool = True):
        """
        Plot the training and validation data residuals

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            The axes to plot on
        log : bool
            If true the y axis is in log scale
        """
        if ax is None:
            f, ax = plt.subplots()
        if log:
            ax.set_yscale("log")
        epochs = self.history["epochs"]
        ax.scatter(epochs, self.history["training loss"],
                   marker=".",
                   label="training loss")
        if any([v is not None for v in self.history["validation loss"]]):
            ax.scatter(epochs, self.history["validation loss"],
                       marker=".",
                       label="validation loss")
        if self.history["best epoch"] is not None:
            ax.axvline(self.history["best epoch"], color="k")
        ax.set_ylabel("loss")
        ax.set_xlabel("epochs")
        ax.legend()
        f.tight_layout()

    @property
    def GPU(self) -> bool:
        """
        Returns the GPU(s) the model is evaluated on

        Returns
        -------
        None or int or list of int :
            The GPU the model is evaluated on
        """
        return list(self._GPU) if isinstance(self._GPU, list) else self._GPU

    @GPU.setter
    def GPU(self, value: Union[None, int, List[int]]):
        """
        Set the GPU(s) the model is evaluated on

        Parameters
        ----------
        value : int, or list of int, or None
            The index of the GPU to use to evaluate the model
            Set to None for evaluating on CPU
            Set to a list of int to evaluate on multi GPUs
        """
        assert (value is None) or (type(value) in [int, list])
        # If CUDA is not available falls back to using CPU
        if (value is not None) and not(torch.cuda.is_available()):
            warnings.warn("CUDA is not available on this computer, "
                          "falling back to evaluating on CPU")
            value = None
        # Check that GPU indices are valid
        if value is not None:
            devices = value if isinstance(value, list) else [value]
            for device in devices:
                assert isinstance(device, int)
                if device >= torch.cuda.device_count():
                    gpus = list(range(torch.cuda.device_count()))
                    warnings.warn(f"GPU {device} is not in the list of "
                                  f"available GPUs: {gpus}. "
                                  "Falling back to evaluating on CPU")
                    value = None
                    break
        # Remove duplicates GPU indices
        if isinstance(value, list):
            value = list(set(value))
        # # If a single element in list, convert list to single int
        # if isinstance(value, list) and len(value) == 1:
        #     value = value[0]
        # Set the GPU
        self._GPU = value
        self.module.to(self.device)

    @property
    def device(self) -> torch.device:
        """Return the torch device the model/data are loaded on"""
        if self.GPU is None:
            return torch.device("cpu")
        elif isinstance(self.GPU, int):
            return torch.device(f"cuda:{self.GPU}")
        elif isinstance(self.GPU, list):
            return torch.device(f"cuda:{self.GPU[0]}")

    @property
    def optimization_method(self) -> str:
        """
        returns the name of the optimization method

        Returns
        -------
        str :
            name of the method
        """
        return self._optimization_method

    @optimization_method.setter
    def optimization_method(self, name: str):
        """
        set the optimization method for training the model.
        must be the name of an optimizer class from 'torch.optim'.

        This also resets the optimization parameters (gradient momentum,
        learning rate decay, ...)

        Parameters
        ----------
        name : str
            the name of the optimization method
        """
        if not hasattr(torch.optim, name):
            available = [n for n in dir(torch.optim) if n[0] != "_"]
            raise ValueError(f"Invalid optimizer '{name}', "
                             f"valid options are: {available}")
        cls = getattr(torch.optim, name)
        self.optimizer = cls(self.module.parameters(), 0.001)
        self._optimization_method = name

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "GPU": self.GPU,
                "optimization method": self.optimization_method,
                "history": self.history,
                "module": self.module.dump}

    def _set_learning_rate(self, lr: float):
        """
        set the learning rate for the training

        Parameters:
        -----------
        lr : float
            new learning rate
        """
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _set_norm_update_factor(self, f: Union[float, None]):
        """
        Set the update factor 'f' used for batch normalization
        moment = f*batch_moment + (1-f)*moment
        Where 'moment' are the mean and variance

        f must be between 0. and 1.
        or None to use an averaging method instead

        Parameters
        ----------
        f : float or None
            the update factor
        """
        assert (f is None) or (0. <= f <= 1.)
        for m in self.module.modules():
            if type(m).__name__.startswith("BatchNorm"):
                m.momentum = f

    def _data_to_tensor(self, X, Y, weights=None,
                        device=torch.device("cpu")) -> tuple:
        """
        Place holder for the method that converts input data to torch tensor
        """
        raise NotImplementedError(f"'_data_to_tensor' not implemented for "
                                  f"model '{type(self)}'")

    def _tensor_to_y(self, tensor: torch.Tensor) -> object:
        """
        Place holder for the method that converts torch tensor to model output
        """
        raise NotImplementedError(f"'_tensor_to_y' not implemented for "
                                  f"model '{type(self)}'")

    def _get_state(self) -> tuple:
        """
        Returns a snapshot of the model's state

        The 'state_dict' are deep copied otherwise the saved tensors are
        modified along with the network's training

        Returns
        -------
        dict :
            the state of the model
        """
        params = self.module.state_dict(keep_vars=True)
        grads = {k: None if t.grad is None else t.grad.tolist()
                 for k, t in params.items()}
        return {"params": {k: t.tolist() for k, t in params.items()},
                "grad": grads,
                "optim": self.optimizer.state_dict()}

    def _set_state(self, state: tuple):
        """
        Loads a snapshot of the model's state, as returned by '_get_state'

        Parameters
        ----------
        state : dict
            The state of the model
        """
        if "params" in state.keys():
            params = {k: torch.tensor(t, device=self.device)
                      for k, t in state["params"].items()}
            self.module.load_state_dict(params)
        if "grad" in state.keys():
            params = self.module.state_dict(keep_vars=True)
            for key in params.keys():
                t = state["grad"][key]
                if t is not None:
                    t = torch.tensor(t, device=self.device)
                params[key].grad = t
        if "optim" in state.keys():
            self.optimizer.load_state_dict(state["optim"])

    def _training_loop(self, loss_module: torch.nn.Module,
                       training_data: Callable,
                       validation_data: Optional[Callable],
                       n_epochs: int,
                       patience: Optional[int],
                       verbose: bool,
                       keep_best: bool,
                       L1: Union[float, None],
                       L2: Union[float, None],
                       best_epoch: int,
                       best_loss: float,
                       best_state: tuple):
        """
        Trains the model for a fixed number of epoch,
        or until validation loss has'nt decreased for 'patience' epochs,
        or until the user press 'ctrl+c'

        At each epoch:
            The parameters are updated using the gradient (0 initially)
            The gradient is set back to 0 (otherwise it accumulates)
            The gradient of the loss is evaluated on the training data
            if validation data are provided:
                The validation loss is evaluated
                if the validation loss is inferior to the previous best:
                    the best loss is updated
                    the state of the model is saved
                otherwise if we waited more than 'patience' epochs:
                    interupts training
        The best state and epoch are then saved

        Parameters:
        -----------
        loss_module : torch.nn.Module
            the module evaluating the loss of the model
        training_data : Callable
            The generator function yielding training batches
        validation_data : Callbale or None
            Same as 'training_data'
            or None if not using validation data
        n_epochs : int
            The number of epochs to perform
        patience : int or None
            The number of epochs waiting for improvement
            before early stoping
        verbose : bool
            If True prints models info while training
        keep_best : bool
            If True, the model is checkpointed at each epoch,
            and the epoch with the lowest loss is loaded back at the end
            (validation loss is used if validation data are provided,
            otherwise, training loss is used)
        L1 : float or None
            The L1 regularization added to the loss function
        L2 : float or None
            The L2 regularization added to the loss function
        best_epoch : int
            the epoch of the previous best state
        best_loss : float
            the value of the previous best validation loss
        best_state : tuple
            the snapshot of the model as returned by 'self._get_state'
        """
        try:
            # looping on epochs
            for epoch in range(best_epoch+1, best_epoch+n_epochs+1):
                msg = f"Epoch {epoch}: "
                self.history["epochs"].append(epoch)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # training loss
                loss = self._eval_loss(loss_module, training_data,
                                       L1, L2, train=True)
                self.history["training loss"].append(loss)
                msg += f"train={loss:.3g}"
                # validation loss
                if validation_data is not None:
                    loss = self._eval_loss(loss_module, validation_data,
                                           L1, L2, train=False)
                    self.history["validation loss"].append(loss)
                    msg += f" val={loss:.3g}"
                else:
                    self.history["validation loss"].append(None)
                # model checkpointing
                if keep_best:
                    if (loss < best_loss):
                        best_epoch = epoch
                        best_loss = loss
                        best_state = self._get_state()
                    else:
                        msg += " - no improvement"
                else:
                    best_epoch = epoch
                # early stoping
                if patience is not None and (epoch - best_epoch) > patience:
                    break
                # message printing
                if verbose:
                    print(msg, flush=True)
        except KeyboardInterrupt:
            if verbose:
                print("Training interrupted by the user", flush=True)
            # Trims data in case user interupted in the midle of the loop
            keys = ["validation loss", "training loss", "epochs"]
            L = min(len(self.history[key]) for key in keys)
            for key in keys:
                self.history[key] = self.history[key][:L]
            best_epoch = min(self.history["epochs"][-1], best_epoch)
        finally:
            # load the best state
            if keep_best:
                self._set_state(best_state)
            # Save the best epoch
            self.history["best epoch"] = best_epoch

    def _eval_loss(self, loss_module: torch.nn.Module,
                   data: Callable, L1: Union[float, None],
                   L2: Union[float, None], train: bool) -> float:
        """
        Evaluates the loss module on the given batches of data.
        If 'train' is True, also backpropagate the gradient.

        If 'train' is True, the computational graph is built.
        (Slower to compute/more memory used)

        Parameters
        ----------
        loss_module : torch.nn.Module
            module evaluating the loss of the model
        data : Callable
            function generator that yields all the (x, y, w) of an epoch
        L1 : float or None
            The L1 regularization added to the loss function
        L2 : float or None
            The L2 regularization added to the loss function
        train : bool
            If True, grad is backpropagated

        Returns
        -------
        torch.Tensor :
            scalar tensor of the evaluated loss
        """
        device = self.device
        losses = []
        if not train:
            # switch to eval mode
            self.module.eval()
        for n, (x, y, w) in enumerate(data()):
            x = self._to(x, device)
            y = self._to(y, device)
            w = self._to(w, device)
            # The '.mean()' is for multi GPU case.
            # The DataParallel module returns a tensor of loss from each GPU
            if train:
                loss = loss_module(x, y, w).mean()
                loss = self._regularization(loss, L1, L2)
                loss.backward()
            else:
                with torch.no_grad():
                    loss = loss_module(x, y, w).mean()
                    # loss = self._regularization(loss, L1, L2)
            losses.append(float(loss))
        if train:
            # Average the gradient
            for p in self.module.parameters():
                p.grad /= (n+1)
        else:
            # switch back to training mode
            self.module.train()
        torch.cuda.empty_cache()
        return sum(losses) / (n+1)

    def _to(self, variable: Union[torch.Tensor, None],
            device: torch.device) -> object:
        """
        Returns variable (X, Y, or weight) stored on the given device.
        Usefull to handle various types of 'variable' that need to be moved
        differently.
        """
        if variable is None:
            return None
        else:
            return variable.to(device)

    def _regularization(self, loss: torch.Tensor,
                        L1: Union[float, None],
                        L2: Union[float, None]) -> torch.Tensor:
        """
        Add L1 and L2 regularization terms to the loss

        Parameters
        ----------
        loss : torch.Tensor
            the scalar tensor representing the loss
        L1 : float or None
            The L1 regularization added to the loss function
        L2 : float or None
            The L2 regularization added to the loss function

        Returns
        -------
        torch.Tensor :
            the regularized loss
        """
        if L1 is not None:
            norm = sum([torch.norm(p, 1)
                        for p in self.module.parameters()])
            loss = loss + L1 * norm
        if L2 is not None:
            norm = sum([torch.norm(p, 2)
                        for p in self.module.parameters()])
            loss = loss + L2 * norm
        return loss

    def _batch_bounds(self, n_data: int, batch_size: Optional[int],
                      n_batches: Optional[int]) -> List[int]:
        """
        returns the bounds segmenting the data in batches
        """
        # both are None
        if (batch_size is None) and (n_batches is None):
            bounds = [0, n_data+1]
        # n_batches is given
        elif (batch_size is None) and (n_batches is not None):
            if n_batches > n_data:
                raise ValueError(f"n_batches={n_batches} "
                                 f"is superior to n_data={n_data}")
            size = n_data/n_batches
            bounds = [int(i*size) for i in range(n_batches+1)]
        # batch_size is given
        elif (batch_size is not None) and (n_batches is None):
            n_batches = math.ceil(n_data/batch_size)
            bounds = [int((i/n_batches)*(n_data))
                      for i in range(n_batches+1)]
        # both batch_size and n_batches are given
        elif (batch_size is not None) and (n_batches is not None):
            prod = n_batches*batch_size
            if prod > n_data:
                raise ValueError(f"n_batches*batch_size = {prod} is "
                                 "superior to the number "
                                 f"of data {n_data}")
            bounds = [i*batch_size for i in range(n_batches+1)]
        return bounds

    def _static_generator(self, X: object, Y: object,
                          W: Optional[object] = None,
                          batch_size: Optional[int] = None,
                          n_batches: Optional[int] = None,
                          device: torch.device = torch.device("cpu"),
                          shuffle: bool = True) -> Callable:
        """
        Returns a generator function that yields the batch of tensors
        The data is casted to tensor only once for efficiency
        """
        n_data = len(X)
        bounds = self._batch_bounds(n_data, batch_size, n_batches)
        X, Y, W = self._data_to_tensor(X, Y, W, device=device)
        if shuffle:
            def generator():
                index = torch.randperm(n_data)
                for (lower, upper) in zip(bounds[:-1], bounds[1:]):
                    x = X[index[lower:upper]] if X is not None else None
                    y = Y[index[lower:upper]] if Y is not None else None
                    w = W[index[lower:upper]] if W is not None else None
                    yield (x, y, w)
        else:
            def generator():
                for (lower, upper) in zip(bounds[:-1], bounds[1:]):
                    x = X[lower:upper] if X is not None else None
                    y = Y[lower:upper] if Y is not None else None
                    w = W[lower:upper] if W is not None else None
                    yield (x, y, w)
        return generator

    def _jit_generator(self, X: object, Y: object,
                       W: Optional[object] = None,
                       batch_size: Optional[int] = None,
                       n_batches: Optional[int] = None,
                       device: torch.device = torch.device("cpu"),
                       shuffle: bool = True) -> Callable:
        """
        Returns a generator function that yields the batch of tensors
        The X input is casted to tensor at each epoch. (jit = Just In Time)
        (for tokenizers with subword)
        """
        n_data = len(X)
        bounds = self._batch_bounds(n_data, batch_size, n_batches)
        _, Yt, Wt = self._data_to_tensor(None, Y, W, device=device)
        if shuffle:
            def generator():
                Xt, _, _ = self._data_to_tensor(X, None, device=device)
                index = torch.randperm(n_data)
                for (lower, upper) in zip(bounds[:-1], bounds[1:]):
                    x = Xt[index[lower:upper]] if Xt is not None else None
                    y = Yt[index[lower:upper]] if Yt is not None else None
                    w = Wt[index[lower:upper]] if Wt is not None else None
                    yield (x, y, w)
        else:
            def generator():
                Xt, _, _ = self._data_to_tensor(X, None, device=device)
                for (lower, upper) in zip(bounds[:-1], bounds[1:]):
                    x = Xt[lower:upper] if Xt is not None else None
                    y = Yt[lower:upper] if Yt is not None else None
                    w = Wt[lower:upper] if Wt is not None else None
                    yield (x, y, w)
        return generator

    def _generator_wrapper(self, generator: Callable, device: torch.device
                           ) -> Callable:
        """
        Wrap a generator function provided by the user,
        converting data to tensor at each batch

        Parameters
        ----------
        generator : Callable
            generator function that yields the (x, y, [weights]) data
            of the model
        device : torch.device
            the device to store the tensor on

        Returns
        -------
        Callable :
            a generator function that yields (x, y, w) tensors/Nonetypes
        """
        def loader():
            for data in generator():
                yield self._data_to_tensor(*data, device=device)
        return loader

    def _as_generator(self, data: Union[Tuple, Callable, None],
                      default_generator: Callable,
                      batch_size: Optional[int], n_batches: Optional[int],
                      device: torch.device, shuffle: bool = True) -> Callable:
        """
        Create a generator function from the training/validation data
        """
        if isinstance(data, Callable):
            return self._generator_wrapper(data)
        elif data is None:
            return None
        else:
            return default_generator(*data, batch_size=batch_size,
                                     n_batches=n_batches, device=device,
                                     shuffle=shuffle)

    def _batch_generator(self, training_data: Tuple,
                         validation_data: Optional[Tuple],
                         batch_size: Optional[int], n_batches: Optional[int],
                         device: torch.device, shuffle: bool = True
                         ) -> Tuple[Callable, Callable]:
        """
        Returns a generator function that yields the data as tensors

        Parameters
        ----------
        training_data : Tuple
            tuple of (x, y, [weights])
        validation_data : Tuple
            tuple of (x, y, [weights]) or None
        batch_size : int or None
            the size of the batches to yield
        n_batches : int or None
            the number of batches to yield
        device : torch.device
            the device to store the tensors on
        shuffle : bool
            if False, the data is not shuffled at each epoch

        Returns
        -------
        Callable :
            a generator function that yields (x, y, w) tensors/Nonetypes
        """
        training = self._as_generator(training_data, self._static_generator,
                                      batch_size, n_batches, device, shuffle)
        val = self._as_generator(validation_data, self._static_generator,
                                 batch_size, n_batches, device, shuffle)
        return training, val
