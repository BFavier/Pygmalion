import math
import torch
import warnings
import matplotlib.pyplot as plt
import torch.nn.parallel as parallel
from typing import Union, Callable, List
from ..model import Model


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
        obj._set_state({"grad": dump["gradient"]})
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

    def train(self, training_data: Union[tuple, Callable],
              validation_data: Union[tuple, Callable, None] = None,
              n_epochs: int = 1000,
              patience: int = 100,
              verbose: bool = True,
              L_minibatchs: Union[int, None] = None):
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
        L_minibatchs : int or None
            Maximum size of the minibatch
            Or None to process the full batch in one go
        """
        self.module.train()
        # Converts training/validation data to tensors
        if not callable(training_data):
            training_data = self._data_to_tensor(*training_data)
        if not callable(validation_data) and validation_data is not None:
            validation_data = self._data_to_tensor(*validation_data)
        # Wrap the module if training on multi GPU
        if isinstance(self.GPU, list):
            module = parallel.DataParallel(self.module, device_ids=self.GPU)
        else:
            module = self.module
        # Initializing
        if self.residuals["best epoch"] is None:
            best_loss = float("inf")
            best_epoch = 0
        else:
            best_epoch = self.residuals["best epoch"]
            i = self.residuals["epochs"].index(best_epoch)
            best_loss = self.residuals["validation loss"][i]
            for v in ["validation loss", "training loss", "epochs"]:
                self.residuals[v] = self.residuals[v][:i+1]
        best_state = self._get_state()
        # trains the model, stops if the user press 'ctrl+c'
        self._training_loop(module,
                            training_data, validation_data, n_epochs,
                            patience, verbose, L_minibatchs,
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
        epochs = self.residuals["epochs"]
        ax.scatter(epochs, self.residuals["training loss"],
                   marker=".",
                   label="training loss")
        ax.scatter(epochs, self.residuals["validation loss"],
                   marker=".",
                   label="validation loss")
        if self.residuals["best epoch"] is not None:
            ax.axvline(self.residuals["best epoch"], color="k")
        if log:
            ax.set_yscale("log")
        ax.set_ylabel("loss")
        ax.set_xlabel("epochs")
        ax.legend()
        f.tight_layout()

    def _loss_function(self, y_pred: torch.Tensor, y_target: torch.Tensor,
                       weights: Union[None, torch.Tensor] = None
                       ) -> torch.Tensor:
        """
        Place holder for the method that calculates the loss function
        """
        raise NotImplementedError(f"'_loss_function' not implemented for "
                                  f"model '{type(self)}'")

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
        x, _, _ = self._data_to_tensor(X, None, None)
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
                    raise ValueError(f"GPU must be one of {gpus}")
        # Remove duplicates GPU indices
        if isinstance(value, list):
            value = list(set(value))
        # If a single element in list, convert list to single int
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
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
                "module": self.module.dump,
                "gradient": self._get_state()["grad"]}

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

    def _training_loop(self, module: torch.nn.Module,
                       training_data: tuple,
                       validation_data: Union[tuple, None],
                       n_epochs: int,
                       patience: int,
                       verbose: bool,
                       L_minibatchs: Union[int, None],
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
        module : torch.nn.Module
            the module of the model
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
        L_minibatchs : int or None
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
                training_loss = self._batch(module,
                                            training_data, L_minibatchs,
                                            train=True)
                if validation_data is not None:
                    validation_loss = self._batch(module,
                                                  validation_data,
                                                  L_minibatchs,
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

    def _batch(self, module: torch.nn.Module,
               data: Union[tuple, Callable], *args, **kwargs) -> float:
        """
        Process the data by batch (or in one go),
        and returns the average loss.

        Parameters
        ----------
        module : torch.nn.Module
            module of the model
        data : tuple or Callable
            The (X, Y, weights) to evaluate the loss on,
            or a function that yields them by batch.
            'X' and 'Y' are tensors, 'weights' is a list of float or None.
        *args : tuple
            The arguments passed to _minibatch
        **kwargs : dict
            The keyword arguments passed to _minibatch

        Returns
        -------
        float :
            The loss function averaged over the batchs
        """
        if not callable(data):
            return self._minibatch(module, data,
                                   *args, **kwargs)
        else:
            losses = []
            for batch_data in data():
                loss = self._minibatch(
                    module,
                    self._data_to_tensor(*batch_data),
                    *args, **kwargs)
                losses.append(loss)
            if len(losses) == 0:
                raise ValueError("The batch iterator returned no batches")
            return sum(losses)/len(losses)

    def _minibatch(self, module: torch.nn.Module,
                   batch_data: tuple,
                   L_minibatchs: Union[int, None],
                   train: bool = False) -> float:
        """
        Process the input batch data in minibatchs
        of maximum size 'L_minibatch', and returns the average loss.

        Parameters
        ----------
        module : torch.nn.Module
            module of the model
        batch_data : tuple
            The (X, Y, weights) to evaluate the loss on.
            'X' and 'Y' are tensors, 'weights' is a list of float or None.
        L_minibatch : int or None
            The maximum number of observations in a minibatch.
            If None the whole data is processed in one go.
        train : bool
            If True, the gradient is back propagated

        Returns
        -------
        float :
            The loss function averaged over the minibatchs
        """
        if L_minibatchs is None:
            return self._eval_loss(module,
                                   *batch_data, train=train)
        else:
            X, Y, weights = self._shuffle(batch_data)
            L_minibatchs = min(max(1, L_minibatchs), len(X))
            n = math.ceil(len(X)/L_minibatchs)
            bounds = [int(i*len(X)/n) for i in range(n+1)]
            losses = []
            for (start, end) in zip(bounds[:-1], bounds[1:]):
                x = X[start:end]
                y = Y[start:end]
                w = None if weights is None else weights[start:end]
                losses.append(self._eval_loss(module,
                                              x, y, w, train=train))
            return sum(losses)/len(losses)

    def _shuffle(self, batch_data):
        """
        Shuffle the data of a batch.
        This is usefull before performing minibatching on it.

        Parameters
        ----------
        batch_data : tuple
            The (X, Y, weights) to evaluate the loss on.
            'X' and 'Y' are tensors, 'weights' is a list of float or None.

        Returns
        -------
        shuffle_batch : tuple
            The tuple of (X, Y, weights) shuffled
        """
        X, Y, weights = batch_data
        p = torch.randperm(len(X))
        X = X[p]
        Y = Y[p]
        weights = weights[p] if weights is not None else None
        return (X, Y, weights)

    def _eval_loss(self, module: torch.nn.Module,
                   x: torch.Tensor, y: torch.Tensor,
                   w: Union[List[float], None] = None,
                   train: bool = False) -> torch.Tensor:
        """
        Evaluates the loss on the given input data.

        If 'train' is True, builds the computational graph
        (Slower to compute/more memory used)
        and backpropagate the gradient.

        Parameters
        ----------
        module : torch.nn.Module
            module to evaluate
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
        if train:
            loss = self._loss_function(module(x), y, w)
            loss = self._regularization(loss)
            loss.backward()
        else:
            with torch.no_grad():
                self.module.eval()
                loss = self._loss_function(module(x), y, weights=w)
                self.module.train()
                loss = self._regularization(loss)
        loss = float(loss)
        torch.cuda.empty_cache()
        return loss

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
