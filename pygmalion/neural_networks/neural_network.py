import math
import torch
import warnings
import matplotlib.pyplot as plt
import torch.nn.parallel as parallel
from typing import Union, Callable, List, Tuple
from ..model import Model


class LossModule(torch.nn.Module):
    """
    A wrapper around the module of the model that evaluates the loss
    in the forward pass

    This is required for balanced memory use with multi GPU training
    """

    def __init__(self, model: 'NeuralNetwork'):
        super().__init__()
        self.module = model.module
        self.loss = model._loss_function

    def forward(self, x, y_target, weights=None):
        y_pred = self.module(x)
        return self.loss(y_pred, y_target, weights=weights).unsqueeze(0)


class NeuralNetwork(Model):
    """
    Parameters
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
        obj.residuals = dump["residuals"]
        obj.module = cls.ModuleType.from_dump(dump["module"])
        obj.optimization_method = dump["optimization method"]
        obj.norm_update_factor = dump["norm update factor"]
        obj.GPU = dump["GPU"]
        obj.learning_rate = dump["learning rate"]
        obj.L1 = dump["L1"]
        obj.L2 = dump["L2"]
        return obj

    def __init__(self, *args,
                 GPU: Union[None, int, List[int]] = None,
                 learning_rate: float = 1.0E-3,
                 norm_update_factor: Union[float, None] = 0.1,
                 optimization_method: str = "Adam",
                 shuffle: bool = True,
                 L1: Union[float, None] = None,
                 L2: Union[float, None] = None,
                 **kwargs):
        """
        Parameters
        ----------
        *args : tuple
            The args passed to the constructor of 'self.module'
        GPU : None or int or list of int
            The indice of the GPU to evaluate the model on
            Set None to evaluate on cpu
        learning_rate : float
            The learning rate of the model
        norm_update_factor : float or None
            The update factor used for batch normalization
        optimization_method : str
            The name of the optimization method
        L1 : float or None
            L1 regularization added to the loss function
        L2 : float or None
            L2 regularization added to the loss function
        **kwargs : dict
            The kwargs passed to the constructor of 'self.module'
        """
        self.module = self.ModuleType(*args, **kwargs)
        self.GPU = GPU
        self.optimization_method = optimization_method
        self.learning_rate = learning_rate
        self.norm_update_factor = norm_update_factor
        self.L1 = L1
        self.L2 = L2
        self.residuals = {"training loss": [],
                          "validation loss": [],
                          "epochs": [],
                          "best epoch": None}

    def train(self, training_data: tuple,
              validation_data: Union[tuple, None] = None,
              n_epochs: int = 1000,
              patience: int = 100,
              verbose: bool = True,
              batch_length: Union[int, None] = None):
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
        patience : int
            The number of epochs before early stopping
        verbose : bool
            If True the loss are displayed at each epoch
        batch_length : int or None
            Maximum size of the batchs
            Or None to process the full data in one go
        """
        self.module.train()
        # Converts training/validation data to tensors
        device = self.device if batch_length is None else torch.device("cpu")
        pinned = (self.device != device)
        training_data = self._data_to_tensor(*training_data, device=device,
                                             pinned=pinned)
        if validation_data is not None:
            validation_data = self._data_to_tensor(*validation_data,
                                                   device=device,
                                                   pinned=pinned)
        # Wrap the module if training on multi GPU
        if isinstance(self.GPU, list):
            loss_module = parallel.DataParallel(LossModule(self),
                                                device_ids=self.GPU)
        else:
            loss_module = LossModule(self)
        # Initializing
        if self.residuals["best epoch"] is None:
            best_loss = float("inf")
            best_epoch = 0
        else:
            best_epoch = self.residuals["best epoch"]
            i = self.residuals["epochs"].index(best_epoch)
            best_loss = self.residuals["validation loss"][i]
            if best_loss is None:
                best_loss = float("inf")
            for v in ["validation loss", "training loss", "epochs"]:
                self.residuals[v] = self.residuals[v][:i+1]
        best_state = self._get_state()
        # trains the model, stops if the user press 'ctrl+c'
        self._training_loop(loss_module,
                            training_data, validation_data, n_epochs,
                            patience, verbose, batch_length,
                            best_epoch, best_loss, best_state)

    def plot_residuals(self, ax=None, log: bool = True):
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
        epochs = self.residuals["epochs"]
        ax.scatter(epochs, self.residuals["training loss"],
                   marker=".",
                   label="training loss")
        ax.scatter(epochs, self.residuals["validation loss"],
                   marker=".",
                   label="validation loss")
        if self.residuals["best epoch"] is not None:
            ax.axvline(self.residuals["best epoch"], color="k")
        ax.set_ylabel("loss")
        ax.set_xlabel("epochs")
        ax.legend()
        f.tight_layout()

    def __call__(self, X):
        """
        Returns the model's evaluation on the given input

        Parameters:
        -----------
        X : Any
            The input X of the model.
            it's type depend on the neural network type.
            see 'help(self.module)'

        Returns
        -------
        Any :
            The returned Y value of the model.
            it's type depends on the neural network type.
            see 'help(self.module)'
        """
        self.module.eval()
        x, _, _ = self._data_to_tensor(X, None, device=self.device)
        y = self.module(x)
        return self._tensor_to_y(y)

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
    def learning_rate(self) -> float:
        """
        Returns the learning rate used for training

        Returns
        -------
        float :
            the learning rate
        """
        if hasattr(self, "_learning_rate"):
            return self._learning_rate
        else:
            return 0

    @learning_rate.setter
    def learning_rate(self, lr: float):
        """
        set the learning rate for the training

        Parameters:
        -----------
        lr : float
            new learning rate
        """
        self._learning_rate = lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

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
        self.optimizer = cls(self.module.parameters(), self.learning_rate)
        self._optimization_method = name

    @property
    def norm_update_factor(self):
        """
        Returns the update factor used for batch normalization

        Returns
        -------
        float or None:
            the update factor
        """
        return self._f

    @norm_update_factor.setter
    def norm_update_factor(self, f: Union[float, None]):
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
        self._f = f

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "GPU": self.GPU,
                "learning rate": self.learning_rate,
                "norm update factor": self.norm_update_factor,
                "optimization method": self.optimization_method,
                "L1": self.L1,
                "L2": self.L2,
                "residuals": self.residuals,
                "module": self.module.dump}

    def _loss_function(self, y_pred: torch.Tensor, y_target: torch.Tensor,
                       weights: Union[None, torch.Tensor] = None
                       ) -> torch.Tensor:
        """
        Place holder for the method that calculates the loss function
        """
        raise NotImplementedError(f"'_loss_function' not implemented for "
                                  f"model '{type(self)}'")

    def _data_to_tensor(self, X, Y, weights=None) -> tuple:
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
                       training_data: tuple,
                       validation_data: Union[tuple, None],
                       n_epochs: int,
                       patience: int,
                       verbose: bool,
                       batch_length: Union[int, None],
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
        training_data : tuple or Callable
            The data provided to 'self._batch'
        validation_data : tuple or Callbale or None
            Same as 'training_data' or None if not using early stoping
        n_epochs : int
            The number of epochs to performe
        patience : int
            The number of epochs waiting for improvement
            before early stoping
        verbose : bool
            If True prints models info while training
        batch_length : int or None
            The maximum number of items in a minibatchs
            or None to not to use minibatchs
        best_epoch : int
            the epoch of the previous best state
        best_loss : float
            the value of the previous best validation loss
        best_state : tuple
            the snapshot of the model as returned by 'self._get_state'
        """
        try:
            for epoch in range(best_epoch+1, best_epoch+n_epochs+1):
                self.optimizer.step()
                self.optimizer.zero_grad()
                training_loss = self._batch_loss(loss_module,
                                                 training_data, batch_length,
                                                 train=True)
                if validation_data is not None:
                    validation_loss = self._batch_loss(loss_module,
                                                       validation_data,
                                                       batch_length,
                                                       train=False)
                    if validation_loss < best_loss:
                        best_epoch = epoch
                        best_loss = validation_loss
                        best_state = self._get_state()
                    elif (epoch - best_epoch) > patience:
                        break
                else:
                    best_epoch = epoch
                    validation_loss = None
                self.residuals["training loss"].append(training_loss)
                self.residuals["validation loss"].append(validation_loss)
                self.residuals["epochs"].append(epoch)
                if verbose:
                    msg = f"Epoch {epoch}: train={training_loss:.3g}"
                    if validation_loss is not None:
                        msg += f" val={validation_loss:.3g}"
                        if best_epoch != epoch:
                            msg += " - no improvement"
                    print(msg, flush=True)
        except KeyboardInterrupt:
            if verbose:
                print("Training interrupted by the user", flush=True)
            # Trims data in case user interupted in the midle of the loop
            keys = ["validation loss", "training loss", "epochs"]
            L = min(len(self.residuals[key]) for key in keys)
            for key in keys:
                self.residuals[key] = self.residuals[key][:L]
            best_epoch = min(self.residuals["epochs"][-1], best_epoch)
        finally:
            # load the best state
            if validation_data is not None:
                self._set_state(best_state)
            # Save the best epoch
            self.residuals["best epoch"] = best_epoch

    def _batch_loss(self, loss_module: torch.nn.Module,
                    data: Union[tuple, Callable],
                    batch_length: Union[int, None],
                    train: bool) -> float:
        """
        Compute the loss on the given data, processing it by batchs of maximum
        size 'batch_length'.
        If 'batch_length' is None, process in one batch.

        Parameters
        ----------
        loss_module : torch.nn.Module
            module evaluating the loss of the model
        data : tuple or Callable
            The (X, Y, weights) to evaluate the loss on,
            or a function that yields them by batch.
            'X' and 'Y' are tensors, 'weights' is a list of float or None.
        L_minibatch : int or None
            The maximum number of observations in a minibatch.
            If None the whole data is processed in one go.
        train : bool
            If True, the gradient is back propagated

        Returns
        -------
        float :
            The loss function averaged over the batchs
        """
        if batch_length is None:
            return self._eval_loss(loss_module,
                                   *data, train)
        else:
            X, Y, weights = self._shuffle(data)
            N = self._len(X)
            batch_length = min(max(1, batch_length), N)
            n = math.ceil(N/batch_length)
            bounds = [int(i*N/n) for i in range(n+1)]
            losses = []
            for (start, end) in zip(bounds[:-1], bounds[1:]):
                x = self._index(X, start=start, end=end)
                y = self._index(Y, start=start, end=end)
                w = self._index(weights, start=start, end=end)
                losses.append(self._eval_loss(loss_module,
                                              x, y, w, train))
            return sum(losses)/len(losses)

    def _shuffle(self, batch_data: Tuple[torch.Tensor]):
        """
        Shuffle the data of a batch.
        This is usefull before performing minibatching on it.

        Parameters
        ----------
        batch_data : tuple or tensors
            The (X, Y, weights) to evaluate the loss on.
            'X' and 'Y' are tensors, 'weights' is a tensor or None.

        Returns
        -------
        shuffle_batch : tuple of tensors
            The tuple of (X, Y, weights) shuffled
        """
        X, Y, weights = batch_data
        p = torch.randperm(self._len(X))
        X = self._index(X, at=p)
        Y = self._index(Y, at=p)
        weights = self._index(weights, at=p)
        return (X, Y, weights)

    def _index(self, variable: Union[torch.Tensor, None, tuple],
               at=None, start=None, end=None, step=None):
        """
        Index/slice the given observations of the variable (X, Y, or weight).
        Usefull to handle various types of 'variable' that need to be indexed
        differently.

        Parameters
        ----------
        variable : tuple of tensor, tensor, or None
            the variable to split/index
        at : None, or int, or list of int
            The observation indexes
            If None the variable is sliced
        start : None, or int
            The start of the slice
        end : None, or int
            The end of the slice
        step : None, or int
            The step of the slicing

        Returns
        -------
        object :
            the indexed/sliced variable
        """
        if isinstance(variable, tuple):
            return tuple([self._index(v, at=at, start=start, end=end,
                                      step=step)
                          for v in variable])
        elif variable is None:
            return None
        elif isinstance(variable, torch.Tensor):
            if at is not None:
                return variable[at]
            else:
                return variable[slice(start, end, step)]
        else:
            raise ValueError(f"Unexpect variable type '{type(variable)}'")

    def _len(self, variable: Union[torch.Tensor, tuple]) -> int:
        """
        Returns the number of observations in the variable (X, Y, or weight).
        Usefull to handle various types of 'variable' that need to be indexed
        differently.
        """
        if isinstance(variable, torch.Tensor):
            return len(variable)
        elif isinstance(variable, tuple):
            return len(variable[0])
        else:
            raise ValueError(f"Unexpect variable type '{type(variable)}'")

    def _eval_loss(self, loss_module: torch.nn.Module,
                   x: torch.Tensor, y: torch.Tensor,
                   w: Union[List[float], None] = None,
                   train: bool = False) -> torch.Tensor:
        """
        Evaluates the loss module on the given batch of data.
        If 'train' is True, also backpropagate the gradient.

        If 'train' is True, the computational graph is built.
        (Slower to compute/more memory used)

        Parameters
        ----------
        loss_module : torch.nn.Module
            module evaluating the loss of the model
        x : torch.Tensor
            observations
        y : torch.Tensor
            target
        w : List of float, or None
            weights
        train : bool
            If True, grad is backpropagated

        Returns
        -------
        float :
            scalar tensor of the evaluated loss
        """
        device = self.device
        x = self._to(x, device)
        y = self._to(y, device)
        w = self._to(w, device)
        if train:
            loss = loss_module(x, y, w).mean()
            loss = self._regularization(loss)
            loss.backward()
        else:
            with torch.no_grad():
                self.module.eval()
                loss = loss_module(x, y, weights=w).mean()
                self.module.train()
                loss = self._regularization(loss)
        loss = float(loss)
        torch.cuda.empty_cache()
        return loss

    def _to(self, variable: Union[torch.Tensor, None, tuple],
            device: torch.device) -> object:
        """
        Returns variable (X, Y, or weight) stored on the given device.
        Usefull to handle various types of 'variable' that need to be moved
        differently.
        """
        if isinstance(variable, torch.Tensor):
            return variable.to(device)
        elif isinstance(variable, tuple):
            return tuple([self._to(v, device) for v in variable])
        elif variable is None:
            return None
        else:
            raise ValueError(f"Unexpect variable type '{type(variable)}'")

    def _regularization(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Add L1 and L2 regularization terms to the loss

        Parameters
        ----------
        loss : torch.Tensor
            the scalar tensor representing the loss

        Returns
        -------
        torch.Tensor :
            the regularized loss
        """
        if self.L1 is not None:
            norm = sum([torch.norm(p, 1)
                        for p in self.module.parameters()])
            loss = loss + self.L1 * norm
        if self.L2 is not None:
            norm = sum([torch.norm(p, 2)
                        for p in self.module.parameters()])
            loss = loss + self.L2 * norm
        return loss
