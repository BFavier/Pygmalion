import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Union, Iterable, Callable, Optional
from .layers._normalizer import Normalizer
from ._conversions import named_to_tensor, floats_to_tensor, tensor_to_floats
from .layers import Dense
from ._neural_network import NeuralNetwork


class ProbabilityDistribution(NeuralNetwork):
    """
    Strongly inspired by this paper :
    Chilinski, P., & Silva, R. (2020, August). Neural likelihoods via cumulative distribution functions. In Conference on Uncertainty in Artificial Intelligence (pp. 420-429). PMLR.
    https://arxiv.org/abs/1811.00974
    """

    def __init__(self, inputs: List[str], hidden_features: list[int],
                 normalize: bool=False, activation: str = "elu", dropout: Optional[float] = None,
                 inf_bound: dict={}, sup_bound: dict={},
                 infinity: float=1.0E10):
        super().__init__()
        self.inputs = list(inputs)
        self.inf = floats_to_tensor([inf_bound.get(i, -infinity) for i in self.inputs]).unsqueeze(0)
        self.sup = floats_to_tensor([sup_bound.get(i, infinity) for i in self.inputs]).unsqueeze(0)
        self.normalizer = Normalizer(-1, len(inputs)) if normalize else None
        self.scaling_weight = torch.nn.Parameter(torch.ones(len(self.inputs)))
        self.scaling_bias = torch.nn.Parameter(torch.zeros(len(self.inputs)))
        self.layers = torch.nn.ModuleList()
        in_features = len(inputs)
        for out_features in hidden_features:
            self.layers.append(Dense(in_features, out_features, normalize=normalize, activation=activation, dropout=dropout))
            in_features = out_features
        self.head = torch.nn.Linear(in_features, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        """
        X = X.to(self.device)
        if self.normalizer is not None:
            X = self.normalizer(X)
        for layer in self.layers:
            X = layer(X)
        return torch.sigmoid(self.head(X)).squeeze(-1)

    def _cdf(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (*, n_inputs)

        Returns
        -------
        torch.Tensor :
            tensor of shape (*)
        """
        return self(X)
        SUP = self(self.inf)
        INF = self(self.sup)
        return (self(X) - INF) / (SUP - INF)
        # return self(X)/self.t(torch.ones(X.shape, device=X.device))
        # INF, SUP = self(self.inf), self(self.sup)
        # return (self(X) - INF) / (SUP - INF)

    def _pdf(self, X: torch.Tensor) -> torch.Tensor:
        """
        returns the pdf defined as d^n/dx0,dx1,...,dxn (cdf)

        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (*, n_inputs)

        Returns
        -------
        torch.Tensor :
            tensor of shape (*)
        """
        X = X.to(self.device)
        n_dims = X.shape[-1]
        f = self._cdf
        for dim in range(n_dims):
            f = self._dfdxi(f, dim)
        return f(X)

    @staticmethod
    def _dfdxi(f: Callable, i: int) -> torch.Tensor:
        """
        Parameters
        ----------
        f : Callable
            function f that takes an input x of shape (*, n_dims),
            and such that y = f(x) is of shape (*)

        Returns
        -------
        Callable :
            returns the derivative of f with regards to the ith dimension of x
        """
        # this copy of 'i' is necessary, because if i is kept as a reference,
        # all derivations will occure along the latest value of i
        ii = int(i)

        def dfdxi(x: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                return torch.autograd.functional.jacobian(lambda x: f(x).sum(), x, create_graph=True)[..., ii]
    
        return dfdxi

    def pdf(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns the probability density function
        """
        self.eval()
        X = self._x_to_tensor(df)
        with torch.no_grad():
            return tensor_to_floats(self._pdf(X))
    
    def cdf(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns the cumulated density function
        """
        self.eval()
        X = self._x_to_tensor(df)
        with torch.no_grad():
            return tensor_to_floats(self._cdf(X))
    
    def loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        cdf loss

        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (*, n_inputs)
        """
        Y = X.to(self.device)
        shape = Y.shape
        Y = Y.reshape(-1, shape[-1])
        N, _ = Y.shape
        Y = (Y.unsqueeze(0) <= Y.unsqueeze(1)).all(dim=-1)  # matrix of shape (N, N) of whether a row observation is inferior to a column observation
        Y = (Y.sum(dim=-1) - 1) / (N-1)
        CDF = (Y.reshape(shape[:-1]) - self._cdf(X))**2
        return torch.mean(CDF)
        PDF = torch.log(1 + torch.exp(self._pdf(X)))
        return torch.mean(CDF/PDF)

    def _x_to_tensor(self, df: Union[pd.DataFrame, dict, Iterable]):
        return named_to_tensor(df, self.inputs)
    
    def data_to_tensor(self, df: Union[pd.DataFrame, dict, Iterable]):
        return (self._x_to_tensor(df),)

    @property
    def device(self) -> torch.device:
        return self.head.weight.device
