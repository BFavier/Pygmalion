import torch
import pandas as pd
import numpy as np
from typing import List, Iterable, Union, Tuple


def floats_to_tensor(arr: Iterable, device: torch.device) -> torch.Tensor:
    """converts an array of numerical values to a tensor of floats"""
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    return torch.tensor(arr, dtype=torch.float, device=device,
                        requires_grad=False)


def tensor_to_floats(tensor: torch.Tensor) -> np.ndarray:
    """converts a torch.Tensor to a numpy.ndarray of doubles"""
    assert tensor.dtype == torch.float
    return tensor.detach().cpu().numpy().astype(np.float64)


def longs_to_tensor(arr: Iterable, device: torch.device) -> torch.Tensor:
    """converts an array of numerical values to a tensor of longs"""
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    return torch.tensor(arr, dtype=torch.long, device=device,
                        requires_grad=False)


def tensor_to_longs(tensor: torch.Tensor) -> list:
    """converts an array of numerical values to a tensor of longs"""
    assert tensor.dtype == torch.long
    return tensor.detach().cpu().numpy()


def images_to_tensor(images: Iterable[np.ndarray],
                     device: torch.device) -> torch.Tensor:
    """Converts a list of images to a tensor"""
    if not isinstance(images, np.ndarray):
        images = np.array(images)
    # Numpy images are of shape (height, width, channel) or (height, width)
    # but pytorch expects (channels, height width)
    if len(images.shape) == 3:  # Grayscale images
        images = np.expand_dims(images, 1)
    else:
        images = np.moveaxis(images, -1, 1)
    return floats_to_tensor(images, device)


def tensor_to_images(tensor: torch.Tensor,
                     colors: Union[np.ndarray, None] = None
                     ) -> np.ndarray:
    """
    Converts a tensor of long to a list of images
    If 'colors' is not None, tensor must contain indexes to the
    color for each pixel.
    Otherwise it must be a tensor of float valued images between 0. and 255.
    """
    if colors is None:
        arr = np.round(tensor_to_floats(tensor))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[1] == 1:  # grayscale images
            return arr[:, 0, :, :]
        elif arr.shape[1] in [3, 4]:  # RGB or RGBA image
            return np.moveaxis(arr, 1, -1)
        else:
            raise ValueError(f"Unexpected number of channels {tensor.shape[1]}"
                             " for tensor representing a list of images")
    else:
        assert tensor.dtype == torch.long
        return colors[tensor_to_longs(tensor)]


def tensor_to_index(tensor: torch.tensor) -> np.ndarray:
    """Converts a tensor to an array of category index"""
    return tensor_to_longs(torch.argmax(tensor, dim=1))


def classes_to_tensor(input: Iterable[str],
                      classes: List[str],
                      device: torch.device) -> torch.Tensor:
    """
    converts a list of classes to tensor
    'classes' must be a list of unique possible classes.
    The tensor contains for each input the index of the category.
    """
    assert isinstance(classes, list)
    return longs_to_tensor([classes.index(i) for i in input], device=device)


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


def bounding_boxes_to_tensor(bboxes: List[dict], image_size: Tuple[int, int],
                             cell_size: Tuple[int, int], classes: List[str],
                             device: torch.device
                             ) -> Tuple[torch.Tensor]:
    """
    Converts a list of bounding boxes to 4 tensors
    (multiple bounding boxes for each image)
    * The 'boxe_size', a tensor of shape (N, 4, H, W) of (x, y, width, height)
      of the bounding boxe falling in the given grid cell.
      (0, 0, 0, 0) bu default for empty cells
    * The 'object_mask', a tensor of shape (N, H, W) which is True when there
      is an object in the grid cell and False otherwise
    * The 'class_index', a tensor of shape (N, H, W) of class indexes present
      in each grid cell.
      0 by default for empty cells
    * The 'weights", either None or a tensor of shape (N, H, W) of floats.
      In which case the value are the weighting in the loss function of each
      object to detect.

    Parameters
    ----------
    bboxes : list of dict
        A dict for each image with the keys
            * x1, x2, y1, y2 : list of int
                the x/y positions of the two corners of the boxe (in pixels)
            * class : list of str
                the classes of the detected objects
            * [weights : optional, list of floats]
                the weighting of each object in the loss function
    image_size : tupe of int
        the (height, width) of the images
    cell_size : tuple of int
        the (height, width) of the anchor grid's cells (in pixels)
    classes : list of str
        list of the unique class names
    device : torch.device
        the device the tensors should be stored on

    Return
    ------
    tuple of torch.Tensor :
        the (boxe_size, object_mask, class_index, weights) tuple of tensors
    """
    h_image, w_image = image_size
    h_cell, w_cell = cell_size
    h_grid, w_grid = h_image // h_cell, w_image // w_cell
    object_mask = np.zeros((len(bboxes), h_grid, w_grid), dtype=np.float)
    class_index = np.zeros((len(bboxes), h_grid, w_grid), dtype=np.long)
    boxe_size = np.zeros((len(bboxes), 4, h_grid, w_grid), dtype=np.float)
    data = [(img, int(0.5*(y1+y2)/h_cell), int(0.5*(x1+x2)/w_cell),
             0.5*(y1+y2)/h_cell % 1, 0.5*(x1+x2)/w_cell % 1,
             abs(y1-y2)/h_cell, abs(x1-x2)/w_cell, classes.index(c))
            for img, bb in enumerate(bboxes)
            for x1, x2, y1, y2, c
            in zip(bb["x1"], bb["x2"], bb["y1"], bb["y2"], bb["class"])]
    img, row, column, y, x, h, w, c = zip(*data)
    object_mask[img, row, column] = 1
    boxe_size[img, :, row, column] = list(zip(x, y, w, h))
    class_index[img, row, column] = c
    weights = [bb.get("weights", None) for bb in bboxes]
    if not(None in weights):
        weights = sum(weights, [])
        cell_weights = np.zeros((len(bboxes), h, w), dtype=np.float)
        cell_weights[img, row, column] = weights
    else:
        cell_weights = None
    boxe_size = torch.tensor(boxe_size, dtype=torch.float, device=device)
    object_mask = torch.tensor(object_mask, dtype=torch.float, device=device)
    class_index = torch.tensor(class_index, dtype=torch.long, device=device)
    if cell_weights is not None:
        cell_weights = torch.tensor(cell_weights, dtype=torch.float,
                                    device=device)
    return boxe_size, object_mask, class_index, cell_weights
