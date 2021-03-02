import torch
import torch.nn.functional as F
from typing import Union, Tuple


def MSE(y_pred: torch.Tensor, y_target: torch.Tensor,
        weights: Union[None, torch.Tensor] = None,
        target_norm: Union[None, torch.nn.Module] = None) -> torch.Tensor:
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
    target_norm : torch.nn.Module
        A BatchNormNd module applied to the y_target
        before calculating the loss

    Returns
    -------
    torch.Tensor :
        the scalar value of the loss
    """
    if target_norm is not None:
        y_target = target_norm(y_target)
    if weights is None:
        return F.mse_loss(y_pred, y_target)
    else:
        return torch.mean(weights * (y_pred - y_target)**2)


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
        The individual observation weights (ignored if None)
    class_weights : None or torch.Tensor
        If None, all classes are equally weighted
        The class-wise weights (ignored if None)

    Returns
    -------
    torch.Tensor :
        the scalar value of the loss
    """
    if weights is None:
        return F.cross_entropy(y_pred, y_target, weight=class_weights)
    else:
        return torch.mean(F.nll_loss(F.log_softmax(y_pred, dim=1), y_target,
                                     weight=class_weights, reduction="none"
                                     ) * weights)


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
    assert (weights is None) and (class_weights is None), "Not implemented yet"
    n_classes = y_pred.shape[1]
    pred = F.softmax(y_pred, dim=1)
    eps = 1.0E-5
    target = F.one_hot(y_target, num_classes=n_classes).permute(0, 3, 1, 2)
    intersect = torch.sum(pred * target, dim=[2, 3])
    cardinality = torch.sum(pred + target, dim=[2, 3])
    dice_coeff = (2.*intersect + eps) / (cardinality + eps)
    loss = 1. - torch.mean(dice_coeff)
    return loss


def object_detector_loss(y_pred: torch.Tensor, y_target: Tuple[torch.Tensor],
                         weights: Union[None, torch.Tensor] = None,
                         class_weights: Union[None, torch.Tensor] = None,
                         target_norm: Union[None, torch.nn.Module] = None
                         ) -> torch.Tensor:
    """
    Parameters
    ----------
    y_pred : tuple of torch.Tensor
        The predictions of the model
        A tuple of (boxe_size, object_proba, class_proba) Where
        * 'boxe_size' is a tensor of [x, y, width, height] of the bboxe
        * 'object_proba' is the probability object presence in the bboxe
        * 'class_proba' is the probability of the object beeing of each class
        Each tensor has the shape (N, B, C, H, W) with
        * N number of observations
        * B number of bounding boxes predicted per grid cell
        * C number of channels (specific for each tensor)
        * H the height of the cell grid
        * W the width of the cell grid
    y_target : torch.Tensor
        The target bounding boxes to predict
        Similar to y_pred except that each tensor is of shape (N, C, H, W)

    Returns
    -------
    torch.Tensor :
        a scalar tensor representing the loss function
    """
    boxe_pred, object_pred, class_pred = y_pred
    boxe_target, object_target, class_target = y_target
    # select the bounding boxe with highest confidence in each cell
    object_pred, indexes = torch.max(object_pred, dim=1)
    # index the highest confidence boxe with some indexing black magic
    indexes = indexes.unsqueeze(1)
    boxe_pred = torch.gather(boxe_pred, 1,
                             indexes.expand((-1, 4, -1, -1)).unsqueeze(1)
                             ).squeeze(1)
    _, _, n, _, _ = class_pred.shape
    class_pred = torch.gather(class_pred, 1,
                              indexes.expand((-1, n, -1, -1)).unsqueeze(1)
                              ).squeeze(1)
    # Calculating the loss part linked to bounding boxe position/size
    weights = object_target if weights is None else object_target*weights
    boxe_loss = MSE(boxe_pred, boxe_target,
                    weights=weights.unsqueeze(1).expand(-1, 4, -1, -1),
                    target_norm=target_norm)
    # Calculate the loss part linked to object presence
    frac = object_target.sum()/len(object_target.view(-1))
    bce_class_weight = torch.stack([frac, 1-frac])*2
    bce_weight = bce_class_weight[object_target.view(-1).long()
                                  ].view_as(object_target)
    object_loss = F.binary_cross_entropy(object_pred, object_target,
                                         weight=bce_weight)
    # Calculate the loss part linked to detected class
    class_loss = cross_entropy(class_pred, class_target,
                               weights=weights, class_weights=class_weights)
    # scaling factor to account for number of boxes per image
    scale = object_target.view(-1).shape[0] / (object_target.sum() + 1.0E-5)
    # Returning the final loss
    return (boxe_loss + object_loss + class_loss)*scale
