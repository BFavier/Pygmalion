import torch
import pandas as pd
import numpy as np
from typing import List, Iterable, Optional, Union
from warnings import warn
from tqdm import tqdm
from pygmalion.tokenizers._utilities import Tokenizer


def named_to_tensor(data: Union[pd.DataFrame, dict, Iterable],
                    names: List[str], device: Optional[torch.device]=None
                    ) -> torch.Tensor:
    """converts named variables to tensors"""
    if isinstance(data, dict):
        data = {k: v if hasattr(v, "__iter__") else [v] for k, v in zip(names, (data[n] for n in names))}
        data = pd.DataFrame.from_dict(data)
    if isinstance(data, pd.DataFrame):
        data = data[names].to_numpy()
    data = floats_to_tensor(data, device=device)
    if len(data.shape) == 1:
        data = data.unsqueeze(-1)
    return data


def floats_to_tensor(arr: Iterable, device: Optional[torch.device] = None) -> torch.Tensor:
    """converts an array of numerical values to a tensor of floats"""
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    t = torch.tensor(arr, dtype=torch.float, device=device,
                     requires_grad=False)
    return t


def tensor_to_floats(tensor: torch.Tensor) -> np.ndarray:
    """converts a torch.Tensor to a numpy.ndarray of doubles"""
    assert tensor.dtype == torch.float
    return tensor.detach().cpu().numpy()


def longs_to_tensor(arr: Iterable, device: Optional[torch.device] = None) -> torch.Tensor:
    """converts an array of numerical values to a tensor of longs"""
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    t = torch.tensor(arr, dtype=torch.long, device=device,
                     requires_grad=False)
    return t


def tensor_to_longs(tensor: torch.Tensor) -> list:
    """converts a tensor of longs to numpy"""
    assert tensor.dtype == torch.long
    return tensor.detach().cpu().numpy()


def images_to_tensor(images: Iterable[np.ndarray],
                     device: Optional[torch.device] = None) -> torch.Tensor:
    """Converts a list of images to a tensor"""
    assert ((isinstance(images, np.ndarray) and images.dtype == np.uint8)
            or all(im.dtype == np.uint8 for im in images))
    images = floats_to_tensor(images, device)/255
    if len(images.shape) == 3:  # Grayscale images
        images = images.unsqueeze(1)  # (N, H, W) --> (N, 1, H, W)
    else:  # RGB/RGBA images
        images = images.permute(0, 3, 1, 2)  # (N, H, W, C) --> (N, C, H, W)
    return images


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


def tensor_to_index(tensor: torch.tensor, dim=1) -> np.ndarray:
    """Converts a tensor to an array of category index"""
    return tensor_to_longs(torch.argmax(tensor, dim=dim))


def classes_to_tensor(input: Iterable[Union[str, int]],
                      classes: Iterable[str],
                      device: Optional[torch.device] = None) -> torch.Tensor:
    """
    converts a list of classes to tensor
    'classes' must be a list of unique possible classes.
    The tensor contains for each input the index of the category.
    """
    indexes = {c: i for i, c in enumerate(classes)}
    return longs_to_tensor([indexes.get(c, c) for c in input],
                           device=device)


def tensor_to_classes(tensor: torch.Tensor,
                      classes: List[str]) -> List[str]:
    """Converts a tensor of category indexes to str category"""
    indexes = tensor_to_index(tensor)
    return [classes[i] for i in indexes]


def tensor_to_dataframe(tensor: torch.Tensor,
                        names: List[str]) -> pd.DataFrame:
    """converts a tensor to dataframe with named columns"""
    return pd.DataFrame(data=tensor_to_floats(tensor), columns=names)


def tensor_to_probabilities(tensor: torch.Tensor,
                            classes: List[str]) -> pd.DataFrame:
    """
    Converts the raw output of a classifier neural network
    to a dataframe of class probability for each observation
    """
    return tensor_to_dataframe(torch.softmax(tensor, dim=-1), classes)


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
    masks = np.stack([np.all(images == c, axis=3) for c in colors])
    if not masks.any(axis=0).all():
        raise RuntimeError("Found color associated to no class")
    return longs_to_tensor(np.argmax(masks, axis=0), device)


def strings_to_tensor(strings: Iterable[str],
                      tokenizer: Tokenizer,
                      device: torch.device,
                      max_sequence_length: Optional[int] = None,
                      raise_on_longer_sequences: bool = False,
                      add_start_end_tokens: bool = False,
                      progress_bar: bool = False,
                      **kwargs) -> torch.Tensor:
    """
    converts a list of sentences to tensor

    Parameters
    ----------
    strings : iterable of str
        a list of strings
    tokenizer : Tokenizer
        the tokenizer to segment strings
    device : torch.device
        the device to host the tensor on
    max_sequence_length : int or None
        Sentences are stored in a tensor with one dimension corresponding to
        the max token sequence's length, and strings shorter are padded.
        If max_sequence_length is specified, the size of this dimension is
        fixed and strings that are longer are droped.
        Otherwise this dimension is defined by the longest encoded string.
    raise_on_longer_sequences : bool
        If True, raise a ValueError if any sequence
        is longer than max_sequence_length once tokenized
    add_start_end_token : bool
        If True, the <START> and <END> special token are appended around each
        encoded string, before padding with <PAD>
    **kwargs : dict
        dict of kwargs passed to the tokenizer when encoding


    Returns
    -------
    torch.Tensor :
        a tensor of shape (N, L) of longs, where:
        * N is the number of strings
        * L is the length of longest sentence
        and each scalar is the index of a word in the lexicon
    """
    pad = tokenizer.PAD
    if progress_bar:
        strings = tqdm(strings, "tokenizing", unit_scale=True)
    strings = [tokenizer.encode(s, **kwargs) for s in strings]
    if add_start_end_tokens:
        start, end = tokenizer.START, tokenizer.END
        strings = [[start] + s + [end] for s in strings]
    if max_sequence_length is None:
        L_max = max(len(s) for s in strings)
    else:
        n = sum(1 for s in strings if len(s) > max_sequence_length)
        if n > 0:
            error = f"Found {n:,}/{len(strings):,} sequences with tokenized length superior to {max_sequence_length}".replace(",", " ")
            if raise_on_longer_sequences:
                raise ValueError(error)
            else:
                warn(error)
        strings = [s if len(s) <= max_sequence_length else [] for s in strings]
        L_max = max_sequence_length
    data = [s + [pad]*(L_max - len(s)) for s in strings]
    return longs_to_tensor(data, device)


def tensor_to_strings(tensor: torch.Tensor, tokenizer: Tokenizer) -> List[str]:
    """
    converts a tensor to a list of sentences

    Parameters
    ----------
    tensor : torch.Tensor
        a tensor of shape (N, L) where:
        * N is the number of sentences
        * L is the length of longest sentence
    tokenizer : Tokenizer
        a tokenizer with a 'decode' method

    Returns
    -------
    list of str :
        a list of sentences,
        each sentence is a set of words separated by whitespaces
    """
    sentences = tensor_to_longs(tensor)[:, 1:-1]
    return [tokenizer.decode(s) for s in sentences]
