import math
import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union, Callable, Type, List, Iterable
from collections import OrderedDict
from .conversions import tensor_to_index, tensor_to_probabilities


def neural_network(cls: torch.nn.Module) -> Type:
    """A decorator that wraps a 'torch.nn.Module' to hide complexity"""

    class NeuralNetwork:
        """
        Parameters
        ----------
        module : torch.nn.Module
            the underlying torch module of the model
        optimizer : torch.optim
            the optimizer used for training
        """
        def __init__(self, *args,
                     GPU: bool = False,
                     learning_rate: float = 1.0E-3,
                     norm_update_factor: Union[float, None] = 0.1,
                     optimization_method: str = "Adam",
                     L1: Union[float, None] = None,
                     L2: Union[float, None] = None,
                     **kwargs):
            """
            Parameters
            ----------
            *args : tuple
                The args passed to the constructor of 'self.module'
            GPU : bool
                If True, calculation on GPU is enabled
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
            self.module = cls(*args, **kwargs)
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

        def fit(self, training_data: Union[tuple, Callable],
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
                training_data = self.module.data_to_tensor(*training_data)
            if not callable(validation_data) and validation_data is not None:
                validation_data = self.module.data_to_tensor(*validation_data)
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
            self._training_loop(training_data, validation_data, n_epochs,
                                patience, verbose, L_minibatchs, best_epoch,
                                best_loss, best_state)

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
            x, _, _ = self.module.data_to_tensor(X, None, None)
            y = self.module(x)
            return self.module.tensor_to_y(y)

        @property
        def GPU(self) -> bool:
            """
            Returns whether the model is evaluated/trained on GPU

            Returns
            -------
            bool :
                whether the GPU mode is enabled
            """
            return self._GPU

        @GPU.setter
        def GPU(self, enable: bool):
            """
            Set whether the model should be evaluated/trained on GPU.
            This is only available with a CUDA capable GPU.

            Returns
            -------
            bool :
                whether the GPU mode is enabled
            """
            if hasattr(self, "_GPU") and self._GPU == enable:
                return
            if enable and not(torch.cuda.is_available()):
                raise RuntimeError("CUDA is not available on this computer, \
                                    can't train on GPU")
            self._GPU = enable
            if self.GPU:
                self.module.device = torch.device("cuda:0")
            else:
                self.module.device = torch.device("cpu")
            self.module.to(self.module.device)

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

        def _get_state(self) -> tuple:
            """
            Returns a snapshot of the model's state

            The 'state_dict' are deep copied otherwise the saved tensors are
            modified along with the network's training

            Returns
            -------
            tuple :
                the tuple of (module, grad, optimizer) state dicts
            """
            module_state = copy.deepcopy(self.module.state_dict())
            params = self.module.state_dict(keep_vars=True)
            grad_state = OrderedDict([(key, self._copy_grad(params[key].grad))
                                      for key in params])
            optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            return module_state, grad_state, optimizer_state

        def _copy_grad(self, grad: Union[torch.Tensor, None]
                       ) -> Union[torch.Tensor, None]:
            """returns a copy of the given grad tensor"""
            return grad if grad is None else grad.detach().clone()

        def _set_state(self, state: tuple):
            """
            Loads a snapshot of the model's state, as returned by '_get_state'

            Parameters
            ----------
            state : tuple
                The tuple of (module.state_dict(), optimizer.state_dict())
            """
            module_state, grad_state, optimizer_state = state
            self.module.load_state_dict(module_state)
            state = self.module.state_dict(keep_vars=True)
            for key in module_state.keys():
                state[key].grad = grad_state[key]
            self.optimizer.load_state_dict(optimizer_state)

        def _training_loop(self, training_data: tuple,
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
                    training_loss = self._batch(training_data, L_minibatchs,
                                                train=True)
                    if validation_data is not None:
                        validation_loss = self._batch(validation_data, L_minibatchs,
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
                    print("Training interupted by the user", flush=True)
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

        def _batch(self, data: Union[tuple, Callable],
                   *args, **kwargs) -> float:
            """
            Process the data by batch (or in one go),
            and returns the average loss.

            Parameters
            ----------
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
                return self._minibatch(data, *args, **kwargs)
            else:
                losses = []
                for batch_data in data():
                    loss = self._minibatch(self.module.data_to_tensor(*batch_data),
                                           *args, **kwargs)
                    losses.append(loss)
                return sum(losses)/len(losses)

        def _minibatch(self, batch_data: tuple,
                       L_minibatchs: Union[int, None],
                       train: bool = False) -> float:
            """
            Process the input batch data in minibatchs
            of maximum size 'L_minibatch', and returns the average loss.

            Parameters
            ----------
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
                return self._eval_loss(*batch_data, train=train)
            else:
                X, Y, weights = batch_data
                L_minibatchs = min(max(1, L_minibatchs), len(X))
                n = math.ceil(len(X)/L_minibatchs)
                bounds = [int(i*len(X)/n) for i in range(n+1)]
                losses = []
                for (start, end) in zip(bounds[:-1], bounds[1:]):
                    x = X[start:end]
                    y = Y[start:end]
                    w = None if weights is None else weights[start:end]
                    losses.append(self._eval_loss(x, y, w, train=train))
                return sum(losses)/len(losses)

        def _eval_loss(self, x: torch.Tensor, y: torch.Tensor,
                       w: Union[List[float], None] = None,
                       train: bool = False) -> torch.Tensor:
            """
            Evaluates the loss on the given input data.

            If 'train' is True, builds the computational graph
            (Slower to compute/more memory used)
            and backpropagate the gradient.

            Parameters
            ----------
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
                loss = self.module.loss(self.module(x), y, w)
                loss = self._regularization(loss)
                loss.backward()
            else:
                with torch.no_grad():
                    self.module.eval()
                    loss = self.module.loss(self.module(x), y, w)
                    loss = self._regularization(loss)
                    self.module.train()
            torch.cuda.empty_cache()
            return float(loss)

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

    return NeuralNetwork


def nn_classifier(cls: Type) -> Type:
    """A decorator that wraps a classifier 'torch.nn.Module'"""

    class NeuralNetworkClassifier(neural_network(cls)):

        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def index(self, X) -> np.ndarray:
            """
            Returns the category index for each observation

            Parameters
            ----------
            X : Any
                The input X of the model.
                it's type depend on the neural network type.
                see 'help(self.module)'

            Returns
            -------
            np.ndarray :
                An array of long, indexes of the 'self.category'
            """
            x, _, _ = self.module.data_to_tensor(X, None, None)
            return tensor_to_index(self.module(x))

        def probability(self, X) -> pd.DataFrame:
            """
            Return the category probabilities for each observations

            Parameters
            ----------
            X : Any
                The input X of the model.
                it's type depend on the neural network type.
                see 'help(self.module)'

            Returns
            -------
            pd.DataFrame :
                DataFrame of category probabilities
                where each column corresponds to a category
            """
            x, _, _ = self.module.data_to_tensor(X, None, None)
            return tensor_to_probabilities(self.module(x))

        @property
        def class_weights(self):
            if self.module.class_weights is None:
                return None
            else:
                return self.module.class_weights.tolist()

        @class_weights.setter
        def class_weights(self, other: Iterable[float]):
            if other is not None:
                assert len(other) == len(self.categories)
                other = torch.tensor(other, dtype=torch.float,
                                     device=self.device)
            self.module.class_weights = other

        @property
        def categories(self):
            return self.module.categories

    return NeuralNetworkClassifier
