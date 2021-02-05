import torch
import pandas as pd
import numpy as np
from typing import List, Iterable, Union


def floats_to_tensor(arr: Iterable, device: torch.device) -> torch.Tensor:
    """converts an array of numerical values to a tensor of floats"""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return torch.tensor(arr, dtype=torch.float, device=device,
                        requires_grad=False)


def tensor_to_floats(tensor: torch.Tensor) -> np.ndarray:
    """converts a torch.Tensor to a numpy.ndarray of doubles"""
    assert tensor.dtype == torch.float
    return tensor.detach().cpu().numpy().astype(np.float64)


def longs_to_tensor(arr: Iterable, device: torch.device) -> torch.Tensor:
    """converts an array of numerical values to a tensor of longs"""
    return torch.tensor(arr, dtype=torch.long, device=device,
                        requires_grad=False)


def tensor_to_longs(tensor: torch.Tensor) -> list:
    """converts an array of numerical values to a tensor of longs"""
    assert tensor.dtype == torch.long
    return tensor.detach().cpu().numpy()


def images_to_tensor(images: Iterable[np.ndarray],
                     device: torch.device) -> torch.Tensor:
    """Converts a list of images to a tensor"""
    assert all(im.dtype == np.uint8 for im in images)
    # Numpy images are of shape (height, width, channel) or (height, width)
    # but pytorch expects (channels, height width)
    images = [np.moveaxis(im if len(im.shape) == 3 else im[:, :, None],
                          -1, 0).astype(float)
              for im in images]
    return floats_to_tensor(images, device)


def tensor_to_images(tensor: torch.Tensor,
                     colors: Union[np.ndarray, None] = None
                     ) -> np.ndarray:
    """
    Converts a tensor of long to a list of images
    If 'colors' is not None, tensor must contain indexes to the
    color for each pixel.
    Otherwise it must be a tensor of float valued images between 0 and 255.
    """
    if colors is None:
        assert tensor.dtype == torch.float
        arr = np.round(tensor_to_floats(tensor))  # round results
        arr = np.clip(arr, 0, 255).astype("uint8")  # converts to uint8
        if arr.shape[1] == 1:  # grayscale images
            return arr[:, 0, :, :]
        elif arr.shape[1] in [3, 4]:  # RGB or RGBA image
            return np.moveaxis(arr, 1, -1)
    else:
        assert tensor.dtype == torch.long
        return colors[tensor_to_longs(tensor)]


def classes_to_tensor(input: Iterable[str],
                         classes: List[str],
                         device: torch.device) -> torch.Tensor:
    """
    converts a list of categorical variables to tensor
    'classes' must be a list of unique possible classes.
    The tensor contains for each input the index of the category.
    """
    assert isinstance(classes, list)
    return longs_to_tensor([classes.index(i) for i in input], device=device)


def tensor_to_index(tensor: torch.tensor) -> np.ndarray:
    """Converts a tensor to an array of category index"""
    return tensor_to_longs(torch.argmax(tensor, dim=1))


def tensor_to_classes(tensor: torch.Tensor,
                         classes: List[str]) -> List[str]:
    """Converts a tensor of category indexes to str category"""
    indexes = tensor_to_index(tensor)
    return np.array(classes)[indexes]


def dataframe_to_tensor(df: pd.DataFrame,
                        x: List[str],
                        device: torch.device) -> torch.Tensor:
    """Converts a dataframe of numerical values to tensor"""
    assert all(np.issubdtype(df[var].dtype, np.number) for var in x)
    arr = df[x].to_numpy(dtype=np.float32)
    return floats_to_tensor(arr, device=device)


def tensor_to_probabilities(tensor: torch.Tensor,
                            classes: List[str]) -> pd.DataFrame:
    """
    Converts the raw output of a classifier neural network
    to a dataframe of class probability for each observation
    """
    arr = tensor_to_floats(torch.softmax(tensor, dim=-1))
    return pd.DataFrame(data=arr, columns=classes)


def segmented_to_tensor(images: np.ndarray, colors: Iterable,
                        device: torch.device) -> torch.Tensor:
    """
    Converts a segmented image to a tensor of long
    """
    if len(images.shape) == 4:  # Color image
        assert all(hasattr(c, "__iter__") for c in colors)
    elif len(images.shape) == 3:  # Grayscale image
        assert all(isinstance(c, int) for c in colors)
        images = np.expand_dims(images, -1)
        colors = [[c] for c in colors]
    else:
        raise RuntimeError("Unexpected shape of segmented images")
    masks = np.array([np.all(images == c, axis=3) for c in colors])
    if not masks.any(axis=0).all():
        raise RuntimeError("Found color associated to no class")
    return longs_to_tensor(np.argmax(masks, axis=0), device=device)
