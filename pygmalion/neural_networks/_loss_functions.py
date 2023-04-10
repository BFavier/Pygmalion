import torch
import torch.nn.functional as F
from typing import Union, Tuple


def MSE(y_pred: torch.Tensor, y_target: torch.Tensor,
        weights: Union[None, torch.Tensor] = None) -> torch.Tensor:
    """
    Returns the Root Mean Squared Error of the model.
    Each observation can optionnaly be weighted

    Parameters
    ----------
    y_pred : torch.Tensor
        A Tensor of float of shape [N_observations]
        The values predicted by the model for eahc observations
    y_target : torch.Tensor
        A Tensor of float of shape [N_observations]
        The target values to be predicted by the model
    weights : None or torch.Tensor
        If None all observations are equally weighted
        Otherwise the squared error of each observation
        is multiplied by the given factor

    Returns
    -------
    torch.Tensor :
        the scalar value of the loss
    """
    y_target = y_target.to(y_pred.device)
    if weights is None:
        return F.mse_loss(y_pred, y_target)
    else:
        weights = weights.to(y_pred.device)
        return torch.sum(weights * (y_pred - y_target)**2) / weights.sum()


def RMSE(*args, **kwargs):
    """
    Returns the Root Mean Squared Error of the model.

    Parameters
    ----------
    *args, **kwargs :
        similar to MSE

    Returns
    -------
    torch.Tensor :
        the scalar value of the loss
    """
    return torch.sqrt(MSE(*args, **kwargs))


def cross_entropy(y_pred: torch.Tensor, y_target: torch.Tensor,
                  weights: Union[None, torch.Tensor] = None,
                  class_weights: Union[None, torch.Tensor] = None,
                  label_smoothing: float = 0.
                  ) -> torch.Tensor:
    """
    Returns the cross entropy error of the model.
    Each observation and each class be optionnaly weighted

    Parameters
    ----------
    y_pred : torch.Tensor
        A Tensor of float of shape [N_observations, N_classes, ...]
        The probability of each class for eahc observation
    y_target : torch.Tensor
        A Tensor of long of shape [N_observations, 1, ...]
        The index of the class to be predicted
    weights : None or torch.Tensor
        The individual observation weights (ignored if None)
    class_weights : None or torch.Tensor
        If None, all classes are equally weighted
        The class-wise weights (ignored if None)

    Returns
    -------
    torch.Tensor :
        the scalar value of the loss
    """
    y_target = y_target.to(y_pred.device)
    if class_weights is not None:
        class_weights = class_weights.to(y_pred.device)
    if weights is None:
        return F.cross_entropy(y_pred, y_target, weight=class_weights, label_smoothing=label_smoothing)
    else:
        weights = weights.to(y_pred.device)
        return torch.sum(F.cross_entropy(y_pred, y_target, weight=class_weights, label_smoothing=label_smoothing, reduction="none")
                         * weights) / weights.sum()


def soft_dice_loss(y_pred: torch.Tensor, y_target: torch.Tensor,
                   weights: Union[None, torch.Tensor] = None,
                   class_weights: Union[None, torch.Tensor] = None
                   ) -> torch.Tensor:
    """
    A soft Dice loss for segmentation

    Parameters
    ----------
    y_pred : torch.Tensor
        A Tensor of float of shape [N_observations, N_classes, ...]
        The probability of each class for eahc observation
    y_target : torch.Tensor
        A Tensor of long of shape [N_observations, 1, ...]
        The index of the class to be predicted
    weights : None or torch.Tensor
        The individual observation weights (ignored if None)
    class_weights : None or torch.Tensor
        If None, all classes are equally weighted
        The class-wise weights (ignored if None)

    Returns
    -------
    torch.Tensor :
        the scalar value of the loss
    """
    if (weights is not None) or (class_weights is not None):
        raise NotImplementedError("Weighting in soft_dice_loss is not implemented yet")
    y_target = y_target.to(y_pred.device)
    n_classes = y_pred.shape[1]
    pred = F.softmax(y_pred, dim=1)
    eps = 1.0E-5
    target = F.one_hot(y_target, num_classes=n_classes).permute(0, 3, 1, 2)
    intersect = torch.sum(pred * target, dim=[2, 3])
    cardinality = torch.sum(pred + target, dim=[2, 3])
    dice_coeff = (2.*intersect + eps) / (cardinality + eps)
    loss = 1. - torch.mean(dice_coeff)
    return loss
