import pathlib
import json
import h5py
from typing import Union
import pygmalion.neural_networks as nn
import pygmalion.unsupervised.tokenizers as tok


def load(file: str) -> object:
    """
    Load a model from the disk (must be a .json or .h5)

    Parameters
    ----------
    file : str
        path of the file to read
    """
    file = pathlib.Path(file)
    if not file.is_file():
        raise FileNotFoundError("The file '{file}' does not exist")
    suffix = file.suffix.lower()
    if suffix == ".json":
        with open(file, "r", encoding="utf-8") as json_file:
            dump = json.load(json_file)
    elif suffix == ".h5":
        f = h5py.File(file, "r")
        dump = _load_h5(f["dump"])
    else:
        raise ValueError("The file must be '.json' or '.h5' file, "
                         f"but got a '{suffix}'")
    if "type" not in dump.keys():
        raise KeyError("The model's dump doesn't have a 'type' key")
    typename = dump["type"]
    for subpackage in [nn, tok]:
        if hasattr(subpackage, typename):
            cls = getattr(subpackage, typename)
            return cls.from_dump(dump)
    raise ValueError(f"Unknow model type: '{typename}'")


def _load_h5(group: Union[h5py.Group, h5py.Dataset]) -> dict:
    """
    Recursively load the content of an hdf5 file into a python dict.

    Parameters
    ----------
    group : h5py.Group
        An hdf5 group (or the opened file)

    Returns
    -------
    dict
        The model dump as a dict
    """
    group_type = group.attrs["type"]
    if group_type == "dict":
        return {name: _load_h5(group[name]) for name in group}
    elif group_type == "list":
        return [_load_h5(group[name]) for name in group]
    elif group_type == "numbers":
        return group[...].tolist()
    elif group_type == "str":
        return group[...].tolist().decode("utf-8")
    elif group_type == "None":
        return None
    else:
        raise ValueError(f"Unknown group type '{group_type}'")
