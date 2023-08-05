import json
import pathlib
import pickle
import torch
from typing import Union, Type
from io import IOBase
from pygmalion._model import Model
from ._download import download_bytes
from pygmalion.unsupervised import *
from pygmalion.neural_networks import *
from pygmalion.decision_trees import *
from pygmalion.tokenizers import *


def load_model(file_path: Union[str, pathlib.Path, IOBase]) -> Model:
    """
    loads a model from the disk, or download from google drive.

    Parameters
    ----------
    file_path : str or IOBase
        Path to the file to load the model from.
        Can be a path on the disk, an url to download from,
        or an IO stream to read from (the IO stream will be closed after loading).

    Returns
    -------
    Model :
        The loaded model.
    """
    if not isinstance(file_path, IOBase):
        if str(file_path).startswith("https://"):
            file_path = download_bytes(file_path)
        else:
            file_path = pathlib.Path(file_path)
            if not file_path.parent.is_dir():
                raise FileNotFoundError(f"The directory does not exist: '{file_path.parent}'")
            elif not file_path.is_file():
                raise FileNotFoundError(f"The file does not exist or is not a file: '{file_path}'")
    try:
        return torch.load(file_path, map_location="cpu")
    except pickle.UnpicklingError:
        if isinstance(file_path, IOBase):
            file_path.seek(0)
        else:
            file_path = open(file_path, "r", encoding="utf-8")
        dump = json.load(file_path)
        if isinstance(file_path, IOBase):
            file_path.close()
        model_type = dump["type"]
        cls: Type[Model] = globals()[model_type]
        return cls.from_dump(dump)
