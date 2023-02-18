import torch
import copy
from typing import Union, Sequence, Optional, Callable, Iterable
from ._conversions import floats_to_tensor


class NeuralNetwork(torch.nn.Module):
    """
    Abstract class for neural networks
    Implemented as a simple wrapper around torch.nn.Module
    with 'fit' and 'predict' methods
    """

    def __init__(self):
        super().__init__()

    def fit(self, training_data: Iterable,
            validation_data: Optional[Iterable] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            n_steps: int = 1000,
            learning_rate: Union[float, Callable[[int], float]] = 1.0E-3,
            patience: Optional[int] = None,
            keep_best: bool = True,
            loss: Optional[Callable] = None,
            L1: Optional[float] = None,
            L2: Optional[float] = None,
            metric: Optional[Callable] = None,
            verbose: bool = True):
        """
        Trains a neural network model.

        Parameters
        ----------
        training_data : Iterable of (x, y) or (x, y, weights) data
            The data used to fit the model on.
            A tuple of (x, y[, weights]) or a callable that yields them.
            The type of each element depends on the model kind.
        validation_data : None or Iterable of (x, y) or (x, y, weights) data
            The data used to test for early stoping.
            Similar to training_data or None
        optimizer : torch.optim.Optimizer or None
            optimizer to use for training
        n_steps : int
            The maximum number of optimization steps
        learning_rate : float or Callable
            The learning rate used to update the parameters,
            or a learning rate function of the number of optimization steps
            performed
        patience : int or None
            The number of steps before early stopping
            (if no improvement for 'patience' steps, stops training early)
            If None, no early stoping is performed
        keep_best : bool
            If True, the model is checkpointed at each step if there was
            improvement,
            and the best model is loaded back at the end of training
        verbose : bool
            If True the loss are displayed at each epoch
        """
        best_step = 0
        best_state = copy.deepcopy(self.state_dict())
        best_metric = None
        train_losses = []
        val_losses = []
        if optimizer is None:
            lr = learning_rate(0) if callable(learning_rate) else learning_rate
            optimizer = torch.optim.Adam(self.parameters(), lr)
        else:
            pass
        try:
            # looping on epochs
            for step in range(n_steps+1):
                # stepping the optimization
                optimizer.step()
                # updating learning rate
                if callable(learning_rate):
                    for g in optimizer.param_groups:
                        g["lr"] = learning_rate(step)
                optimizer.zero_grad()
                # training loss
                self.train()
                train_loss = []
                for batch in training_data:
                    loss = self.loss(*batch)
                    loss.backward()
                    train_loss.append(loss.item())
                train_loss = sum(train_loss) / max(1, len(train_loss))
                train_losses.append(train_loss)
                # validation data
                self.eval()
                if validation_data is not None:
                    val_loss = []
                    with torch.no_grad():
                        for batch in validation_data:
                            val_loss.append(self.loss(*batch).item())
                    val_loss = sum(val_loss) / max(1, len(val_loss))
                else:
                    val_loss = None
                val_losses.append(val_loss)
                # model checkpointing
                metric = val_loss if val_loss is not None else train_loss
                if best_metric is None or metric < best_metric:
                    best_step = step
                    best_metric = metric
                    if keep_best:
                        best_state = copy.deepcopy(self.state_dict())
                # early stoping
                if patience is not None and (step - best_step) > patience:
                    break
                # message printing
                if verbose:
                    print(f"Step {step}: train loss = {train_loss:.3g}, val loss = {metric}")
        except KeyboardInterrupt:
            if verbose:
                print("Training interrupted by the user")
        finally:
            # load the best state
            if keep_best:
                self.load_state_dict(best_state)
        return train_losses, val_losses, best_step
    
    # def norm(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    #     """
    #     """
    #     return 0

    def data_to_tensor(self, x: object, y: object,
                        weights: Optional[Sequence[float]] = None,
                        device: Optional[torch.device] = None) -> tuple:
        x = self._x_to_tensor(x, device)
        y = self._y_to_tensor(y, device)
        if weights is not None:
            w = floats_to_tensor(weights, device)
            data = (x, y, w/w.mean())
        else:
            data = (x, y)
        return data

    def predict(self, *args):
        self.eval()
        x = self._x_to_tensor(*args)
        with torch.no_grad():
            y_pred = self(x)
        return self._tensor_to_y(y_pred)

    def _x_to_tensor(self, x: object) -> torch.Tensor:
        raise NotImplementedError()

    def _y_to_tensor(self, y: object) -> torch.Tensor:
        raise NotImplementedError()

    def _tensor_to_y(self, T: torch.Tensor) -> object:
        raise NotImplementedError()


class NeuralNetworkClassifier(NeuralNetwork):
    """
    Abstract class for classifier neural networks
    Implement a 'probabilities' method in addition to the 'NeuralNetwork'
    class methods
    """

    def __init__(self):
        super().__init__()
    
    def probabilities(self, *args):
        self.eval()
        x = self._x_to_tensor(*args)
        with torch.no_grad():
            y_pred = self(x)
        return self._tensor_to_proba(y_pred)
    
    def _tensor_to_proba(self, T: torch.Tensor) -> object:
        raise NotImplementedError()
