import torch
import torch.nn.functional as F
from typing import Optional


class Dropout(torch.nn.Module):

    def __init__(self, p: Optional[float]):
        super().__init__()
        self.p = p
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (*)
        
        Returns
        -------
        torch.Tensor :
            tensor of shape (*)
        """
        if self.p is not None:
            return F.dropout(X, self.p, self.training)
        else:
            return X


class Dropout1d(Dropout):

    def __init__(self, p: Optional[float]):
        super().__init__(p)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (N, C, L)
        
        Returns
        -------
        torch.Tensor :
            tensor of shape (N, C, L)
        """
        if self.p is not None:
            return F.dropout1d(X, self.p, self.training)
        else:
            return X


class Dropout2d(Dropout):

    def __init__(self, p: Optional[float]):
        super().__init__(p)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of shape (N, C, H, W)
        
        Returns
        -------
        torch.Tensor :
            tensor of shape (N, C, H, W)
        """
        if self.p is not None:
            return F.dropout2d(X, self.p, self.training)
        else:
            return X
