import pathlib
import json
import h5py
from . import neural_networks as nn
from . import unsupervised as un


def load(file: str) -> object:
    """
    Load a model from the disk (must be a .json or .h5)

    Parameters
    ----------
    file : str
        path of the file to read
    """
    file = pathlib.Path(file)
    suffix = file.suffix.lower()
    if not file.is_file():
        raise FileNotFoundError("The file '{file}' does not exist")
    if suffix == ".json":
        with open(file) as json_file:
            dump = json.load(json_file)
    elif suffix == ".h5":
        f = h5py.File(file, "r")
        dump = _load_h5(f)
    else:
        raise ValueError("The file must be '.json' or '.h5' file, "
                         f"but got a '{suffix}'")
    if "type" not in dump.keys():
        raise KeyError("The model's dump doesn't have a 'type' key")
    typename = dump["type"]
    for subpackage in [nn, un]:
        if hasattr(subpackage, typename):
            cls = getattr(subpackage, typename)
            return cls.from_dump(dump)
    raise ValueError(f"Unknow model type: '{typename}'")


def _load_h5(cls, group: h5py.Group) -> dict:
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
    elif group_type == "scalar":
        return group.attrs["data"].tolist()
    elif group_type == "str":
        return group.attrs["data"]
    elif group_type == "None":
        return None
    elif group_type == "binary":
        return group["data"][...].tolist()
    else:
        raise ValueError(f"Unknown group type '{group_type}'")
