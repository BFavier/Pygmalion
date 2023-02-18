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
        return (F.nll_loss(F.log_softmax(y_pred, dim=1), y_target,
                           weight=class_weights, reduction="none")
                * weights) / weights.mean()


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
    coords_pred, size_pred, object_pred, class_pred = y_pred
    coords_target, size_target, object_mask, class_target = y_target
    _, n, _, _ = class_pred.shape
    # Calculating the loss part linked to bounding boxe position/size
    index = object_mask.unsqueeze(0).expand(2, -1, -1, -1)
    coords_pred = torch.transpose(coords_pred, 0, 1)[index].view(2, -1)
    coords_target = torch.transpose(coords_target, 0, 1)[index].view(2, -1)
    coords_loss = (coords_pred - coords_target)**2
    size_pred = torch.transpose(size_pred, 0, 1)[index].view(2, -1)
    size_target = torch.transpose(size_target, 0, 1)[index].view(2, -1)
    if target_norm is not None:
        size_target = target_norm(size_target.unsqueeze(0)).squeeze(0)
    size_loss = (size_pred - size_target)**2
    if weights is not None:
        coords_loss = coords_loss * weights[object_mask]
        size_loss = size_loss * weights[object_mask]
    coords_loss = coords_loss.mean()
    size_loss = size_loss.mean()
    # Calculate the loss part linked to object presence
    object_loss = -torch.log(object_pred[object_mask] + 1.0E-5)
    non_object_loss = -torch.log(1 - object_pred[~object_mask] + 1.0E-5)
    if weights is not None:
        object_loss = object_loss * weights[object_mask]
        non_object_loss = non_object_loss * weights[~object_mask]
    object_loss = object_loss.mean()
    non_object_loss = non_object_loss.mean()
    # Calculate the loss part linked to detected class
    index = object_mask.unsqueeze(3).expand(-1, -1, -1, n)
    class_pred = class_pred.permute(0, 2, 3, 1)[index].view(-1, n)
    class_target = class_target[object_mask].view(-1)
    class_loss = F.nll_loss(F.log_softmax(class_pred, dim=1), class_target,
                            weight=class_weights, reduction="none")
    if weights is not None:
        class_loss = class_loss * weights[object_mask]
    class_loss = class_loss.mean()
    # Returning the final loss
    return coords_loss + size_loss + object_loss + non_object_loss + class_loss
