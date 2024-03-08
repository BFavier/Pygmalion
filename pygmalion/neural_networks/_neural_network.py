import io
import pathlib
import math
import torch
from typing import Union, Sequence, Optional, Callable, Iterable
from ._conversions import floats_to_tensor
from .layers import Dropout
from pygmalion._model import Model
from datetime import datetime


class NeuralNetwork(torch.nn.Module, Model):
    """
    Abstract class for neural networks
    Implemented as a simple wrapper around torch.nn.Module
    with 'fit' and 'predict' methods
    """

    def __init__(self):
        torch.nn.Module.__init__(self)
        Model.__init__(self)

    def save(self, file_path: Union[str, pathlib.Path, io.IOBase],
             overwrite: bool = False, create_dir: bool = False):
        """
        Saves the model to the disk as '.pth' file

        Parameters
        ----------
        file : str or pathlib.Path or file like
            The path where the file must be created
        overwritte : bool
            If True, the file is overwritten
        create_dir : bool
            If True, the directory to the file's path is created
            if it does not exist already
        """
        if not isinstance(file_path, io.IOBase):
            file_path = pathlib.Path(file_path)
            path = file_path.parent
            suffix = file_path.suffix.lower()
            if suffix != ".pth":
                raise ValueError(
                    f"The model must be saved as a '.pth' file, but got '{suffix}'")
            if not(create_dir) and not path.is_dir():
                raise ValueError(f"The directory '{path}' does not exist")
            else:
                path.mkdir(exist_ok=True)
            if not(overwrite) and file_path.exists():
                raise FileExistsError(
                    f"The file '{file_path}' already exists, set 'overwrite=True' to overwrite.")
            torch.save(self, file_path)
        else:
            torch.save(self, file_path)

    def fit(self, training_data: Union[Iterable, tuple],
            validation_data: Optional[Union[Iterable, tuple]] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            n_steps: int = 1000,
            learning_rate: Union[float, Callable[[int], float]] = 1.0E-3,
            patience: int = math.inf,
            keep_best: bool = True,
            L1: Optional[float] = None,
            L2: Optional[float] = None,
            gradient_cliping: Optional[float] = None,
            step_offset: int = 0,
            backup_path: Optional[str] = None,
            backup_prefix: str = "model",
            backup_frequency: int = 10000,
            verbose: bool = True):
        """
        Trains a neural network model.

        Parameters
        ----------
        training_data : tuple or Iterable of tuples
            The *args passed to the model's loss.
            If an iterable instead, the number of batches yielded makes the number of gradient accumulation steps.
        validation_data : None, or same as training data
            The data used for early stoping.
            Similar to training_data or None
        optimizer : torch.optim.Optimizer or None
            optimizer to use for training
        n_steps : int
            The maximum number of optimization steps
        learning_rate : float or Callable
            The learning rate used to update the parameters,
            or a learning rate function of 'step' the number
            of optimization steps performed
        patience : int or float("inf")
            The number of steps before early stopping
            (if no improvement for 'patience' steps, stops training early)
            If None, no early stoping is performed
        keep_best : bool
            If True, the model is checkpointed at each step if there was
            improvement,
            and the best model is loaded back at the end of training
        L1 : float or None
            L1 regularization factor
        L2 : float
            L2 regularization factor
        gradient_cliping : float or None
            if provided, the gradient vector is cliped in the (-gradient_clipping, gradient_clipping) range
        step_offset : int
            the step to start from, usefull for restarting from a checkpoint with scheduled learning rate
        backup_path : str or path like or None
            if provided, path where to backup the model (and optimizer) on disk
        backup_prefix : str
            prefix of the backup filename (the suffix is the current number step)
        backup_frequency : int
            number of steps before each on-disk backup
        verbose : bool
            If True the loss are displayed at each optimization step
        
        Returns
        -------
        tuple :
            (train_losses, val_losses, grad_norms, best_step)
        """
        if isinstance(training_data, tuple):
            training_data = [training_data]
        if isinstance(validation_data, tuple):
            validation_data = [validation_data]
        if backup_path is not None:
            backup_path = pathlib.Path(backup_path)
            if not backup_path.is_dir():
                raise NotADirectoryError(f"Backup path is not a valid directory: '{backup_path}'")
        best_step = step_offset
        if keep_best:
            best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
        best_metric = None
        train_losses = []
        val_losses = []
        grad_norms = []
        lr = learning_rate(step_offset) if callable(learning_rate) else learning_rate
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr)
        else:
            for g in optimizer.param_groups:
                g["lr"] = lr
        try:
            # looping on epochs
            for step in range(step_offset, step_offset+n_steps+1):
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
                    if L1 is not None:
                        loss = loss + L1 * self._norm(self.parameters(), 1)
                    if L2 is not None:
                        loss = loss + L2 * self._norm(self.parameters(), 2)
                    loss.backward()
                    train_loss.append(loss.item())
                n_batches = len(train_loss)
                train_loss = sum(train_loss) / max(1, n_batches)
                train_losses.append(train_loss)
                # averaging gradient over batches
                if n_batches > 1:
                    for p in self.parameters():
                        if p.grad is not None:
                            p.grad /= n_batches
                # gradient cliping
                if gradient_cliping is not None:
                    for p in self.parameters():
                        if p.grad is not None:
                            p.grad = torch.clip(p.grad, -gradient_cliping, gradient_cliping)
                # gradient norm
                grad_norms.append(self._norm((p.grad for p in self.parameters() if p.grad is not None), 1).item())
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
                        # best_state = copy.deepcopy(self.state_dict())
                        best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                # early stoping
                if (step - best_step) > patience:
                    if verbose:
                        print(f"Early stoping because preformed {patience:,} steps without improvement".replace(",", " "))
                    break
                # message printing
                if verbose:
                    time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                    if val_loss is not None:
                        print(f"{time} Step {step}: train loss = {train_loss:.3g}, val loss = {val_loss:.3g}, grad = {grad_norms[-1]:.3e}")
                    else:
                        print(f"{time} Step {step}: train loss = {train_loss:.3g}, grad = {grad_norms[-1]:.3e}")
                # backup on disk
                if (backup_path is not None) and (step % backup_frequency == 0) and (step > 0):
                    dec = math.floor(math.log10(n_steps)) + 1
                    torch.save(self, backup_path / f"{backup_prefix}_{step:0{dec}}.pth")
                    torch.save(optimizer, backup_path / f"optimizer_{backup_prefix}_{step:0{dec}}.pth")
                    if verbose:
                        print(f"Backed up on disk '{backup_prefix}_{step:0{dec}}.pth'")

        except KeyboardInterrupt:
            if verbose:
                print("Training interrupted by the user")
        finally:
            # load the best state
            if keep_best:
                self.load_state_dict(best_state)
        return train_losses, val_losses, grad_norms, best_step if keep_best else None

    def data_to_tensor(self, x: object, y: object,
                        weights: Optional[Sequence[float]] = None,
                        device: Optional[torch.device] = None,
                        **kwargs) -> tuple:
        x = self._x_to_tensor(x, device, **kwargs)
        y = self._y_to_tensor(y, device, **kwargs)
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
    
    def loss(*args) -> torch.Tensor:
        raise NotImplementedError()
    
    @property
    def dropout(self) -> Optional[float]:
        for m in self.modules():
            if isinstance(m, Dropout):
                return m.p
        return None

    @dropout.setter
    def dropout(self, other: Optional[float]):
        for m in self.modules():
            if isinstance(m, Dropout):
                m.p = other

    def _x_to_tensor(self, x: object) -> torch.Tensor:
        raise NotImplementedError()

    def _y_to_tensor(self, y: object) -> torch.Tensor:
        raise NotImplementedError()

    def _tensor_to_y(self, T: torch.Tensor) -> object:
        raise NotImplementedError()
    
    @staticmethod
    def _norm(tensors: Iterable[torch.Tensor], order: int,
              average: bool=True):
        """
        returns the norm of the tensors
        (normalized by number of elements)
        """
        n, L = 0, 0.
        for t in tensors:
            L = L + torch.sum(torch.abs(t)**order)
            n += t.numel()
        if average:
            L /= n
        return L**(1/order)


class NeuralNetworkClassifier(NeuralNetwork):
    """
    Abstract class for classifier neural networks
    Implement a 'probabilities' method in addition to the 'NeuralNetwork'
    class methods
    """

    def __init__(self, classes: Iterable[str]):
        super().__init__()
        self.classes = tuple(classes)

    def probabilities(self, *args):
        self.eval()
        x = self._x_to_tensor(*args)
        with torch.no_grad():
            y_pred = self(x)
        return self._tensor_to_proba(y_pred)
    
    def _tensor_to_proba(self, T: torch.Tensor) -> object:
        raise NotImplementedError()

    def data_to_tensor(self, x: object, y: object,
                        weights: Optional[Sequence[float]] = None,
                        class_weights: Optional[Sequence[float]] = None,
                        device: Optional[torch.device] = None,
                        **kwargs) -> tuple:
        x = self._x_to_tensor(x, device, **kwargs)
        y = self._y_to_tensor(y, device, **kwargs)
        if weights is not None:
            w = floats_to_tensor(weights, device)
            w = w/w.mean()
        else:
            w = None
        if class_weights is not None:
            wc = floats_to_tensor(class_weights, device)
            wc = wc/wc.mean()
        else:
            wc = None
        if wc is not None:
            data = (x, y, w, wc)
        elif w is not None:
            data = (x, y, w)
        else:
            data = (x, y)
        return data
