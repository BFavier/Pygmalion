import torch
from typing import Union, Callable, Optional
from . import Activation, LayerNorm, Dropout


class Dense(torch.nn.Module):
    """
    A Dense layer is a linear projection followed by optional nomalisation,
    activation, and optional dropout.
    """

    def __init__(self, in_features: int, out_features: int,
                 normalize: bool=True,
                 activation: Union[str, Callable]=torch.relu,
                 dropout: Optional[float]=None,
                 monotonic: bool=False):
        """
        Parameters
        ----------
        in_features : int
            number of input features
        out_features : int
            number of output features
        normalize : bool
            If True, apply layer norm before activation
        activation : str or callable
            Activation function or function name
        dropout : float or None
            dropout probability if any
        monotonic : bool
            If true the model is constrained to be monotonicaly increasing
            with regards to the inputs
        """
        super().__init__()
        self.in_features = in_features
        self.monotonic = monotonic
        self.linear = torch.nn.Linear(in_features, out_features)
        if normalize:
            self.normalization = LayerNorm(-1, out_features)
        else:
            self.normalization = None
        self.activation = Activation(activation)
        self.dropout = Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of floats of shape (*, in_features)
        
        Returns
        -------
        torch.Tensor :
            tensor of shape (*, out_features)
        """
        if self.monotonic:
            X = (torch.abs(self.linear.weight) @ X.unsqueeze(-1)).squeeze(-1)
            if self.linear.bias is not None:
                X = self.linear.bias + X
        else:
            X = self.linear(X)
        if self.normalization is not None:
            X = self.normalization(X)
        X = self.activation(X)
        X = self.dropout(X)
        return X
