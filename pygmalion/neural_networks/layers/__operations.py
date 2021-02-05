import torch
import torch.nn.functional as F


def linear_interpolation(X: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Returns a linear interpolation of the input tensor

    Parameters
    ----------
    X : torch.Tensor
        A tensor of shape (N, C, L)
    size : tuple of int
        The output size (L_out)

    Returns
    -------
    torch.Tensor :
        The interpolated tensor of shape (N, C, L_out)
    """
    assert len(X.shape) == 3
    return F.interpolate(X, size=size, mode="linear")


def bilinear_interpolation(X: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Returns a bilinear interpolation of the input tensor

    Parameters
    ----------
    X : torch.Tensor
        A tensor of shape (N, C, H, W)
    size : tuple of int
        The output size (H_out, W_out)

    Returns
    -------
    torch.Tensor :
        The interpolated tensor of shape (N, C, H_out, W_out)
    """
    assert len(X.shape) == 4
    return F.interpolate(X, size=size, mode="bilinear", align_corners=False)


def overall_pool(X: torch.Tensor, pooling_type: str = "Max"):
    """
    Apply over-all pooling to keep
    """
    dims = X.shape[2:]
    if any(d != 1 for d in dims):
        func = getattr(F, pooling_type.lower()+"_pool2d")
        X = func(X, dims)
    return X.view(X.shape[0], -1)
