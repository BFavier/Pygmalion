import torch
import torch.nn.functional as F
from typing import Union


def RMSE(y_pred: torch.Tensor, y_target: torch.Tensor,
         target_norm: torch.nn.Module,
         weights: Union[None, torch.Tensor] = None) -> torch.Tensor:
    """
    Returns the Root Mean Squared Error of the model.
    Each observation be optionnaly weighted

    Parameters
    ----------
    y_pred : torch.Tensor
        A Tensor of float of shape [N_observations]
        The values predicted by the model for eahc observations
    y_target : torch.Tensor
        A Tensor of float of shape [N_observations]
        The target values to be predicted by the model
    target_norm : torch.nn.Module
        A BatchNormNd module applied to the y_target
        before calculating the loss
    weights : None or torch.Tensor
        If None all observations are equally weighted
        Otherwise the squared error of each observation
        is multiplied by the given factor

    Returns
    -------
    torch.Tensor :
        the scalar value of the loss
    """
    y_target = target_norm(y_target)
    if weights is None:
        return torch.sqrt(F.mse_loss(y_pred, y_target))
    else:
        return torch.sqrt(torch.mean(weights * (y_pred - y_target)**2))


def cross_entropy(y_pred: torch.Tensor, y_target: torch.Tensor,
                  weights: Union[None, torch.Tensor] = None,
                  class_weights: Union[None, torch.Tensor] = None
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
        If None all observations are equally weighted
        Otherwise the squared error of each observation
        is multiplied by the given factor

    Returns
    -------
    torch.Tensor :
        the scalar value of the loss
    """
    if weights is None:
        return F.cross_entropy(y_pred, y_target, weight=class_weights)
    else:
        return F.nll_loss(F.log_softmax(y_pred, dim=1) * weights, y_target,
                          weight=class_weights)


def soft_dice_loss(y_pred: torch.Tensor, y_target: torch.Tensor,
                   weights: Union[None, torch.Tensor] = None,
                   class_weights: Union[None, torch.Tensor] = None):
    """
    A soft Dice loss for segmentation
    """
    # Dice coeff is numerator/denominator
    numerator = 2 * y_pred*y_target
    denominator = y_pred+y_target
    if weights is not None:
        numerator = numerator * weights
        denominator = denominator * weights
    # sum over feature map so that tnesor has shape [N_observations, N_classes]
    N_observations = y_pred.shape[0]
    N_classes = y_pred.shape[1]
    numerator = numerator.view(N_observations, N_classes, -1).sum(dim=-1)
    denominator = denominator.view(N_observations, N_classes, -1).sum(dim=-1)
    # Appy class weighting
    if class_weights is not None:
        numerator = numerator * class_weights
        denominator = denominator * class_weights
    # loss is  (1 - dice coeff) averaged over classes and observations
    dice_coeff = numerator / denominator
    return 1 - torch.mean(dice_coeff, dim=[0, 1])
