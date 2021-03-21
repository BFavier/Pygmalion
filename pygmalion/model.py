import json
import h5py
import pathlib
import numpy as np


class Model:

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        """
        Return an object from a dump
        """
        raise NotImplementedError("Not implemented for class "
                                  f"'{cls.__name__}'")

    def save(self, file: str, overwrite: bool = False):
        """
        Saves a model to the disk (as .json or .h5)

        Parameters
        ----------
        file : str
            The path where the file must be created
        overwritte : bool
            If True, the file is overwritten
        """
        file = pathlib.Path(file)
        path = file.parent
        suffix = file.suffix.lower()
        if not path.is_dir():
            raise ValueError(f"The directory '{path}' does not exist")
        if not(overwrite) and file.exists():
            raise FileExistsError(f"The file '{file}' already exists,"
                                  " set 'overwrite=True' to overwrite.")
        if suffix == ".json":
            with open(file, "w") as json_file:
                json.dump(self.dump, json_file)
        elif suffix == ".h5":
            f = h5py.File(file, "w", track_order=True)
            self._save_h5(f, self.dump)
        else:
            raise ValueError("The model must be saved as a '.json' "
                             f"or '.h5' file, but got '{suffix}'")

    @property
    def dump(self) -> object:
        """Returns a dictionnary representation of the model"""
        raise NotImplementedError("Not implemented for class "
                                  f"'{type(self).__name__}'")

    @classmethod
    def _save_h5(cls, group: h5py.Group, obj: object):
        """
        Recursively populate an hdf5 file with the object

        Parameters
        ----------
        group : h5py.Group
            An hdf5 group (or the opened file)
        obj : object
            The python object to store in the group
        """
        if isinstance(obj, dict):
            group.attrs["type"] = "dict"
            for key, value in obj.items():
                g = group.create_group(key, track_order=True)
                cls._save_h5(g, value)
        elif isinstance(obj, list):
            arr = np.array(obj)
            if np.issubdtype(arr.dtype, np.number):
                group.attrs["type"] = "binary"
                group["data"] = arr
            else:
                group.attrs["type"] = "list"
                for i, value in enumerate(obj):
                    g = group.create_group(f"{i}")
                    cls._save_h5(g, value)
        elif isinstance(obj, str):
            group.attrs["type"] = "str"
            group.attrs["data"] = obj
        elif any(isinstance(obj, t) for t in [float, int, bool]):
            group.attrs["type"] = "scalar"
            group.attrs["data"] = obj
        elif obj is None:
            group.attrs["type"] = "None"
        else:
            raise ValueError(f"Unsupported data type: {type(obj)}")
