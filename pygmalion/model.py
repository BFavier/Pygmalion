import json
import h5py
import pathlib
import numpy as np


class Model:

    @classmethod
    def load(cls, file: str) -> object:
        """
        Load a model from the disk

        Parameters
        ----------
        file : str
            path of the file to read
        """
        file = pathlib.Path(file)
        if not file.is_file():
            raise FileNotFoundError("The file '{file}' does not exist")
        with open(file) as json_file:
            dump = json.load(json_file)
        return cls.from_dump(dump)

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
            raise FileExistsError("The file '{file}' already exists,"
                                  " set 'overwrite=True' to overwrite.")
        if suffix == ".json":
            with open(file, "w") as json_file:
                json.dump(self.dump, json_file)
        elif suffix == ".h5":
            f = h5py.File(file, "w")
            self._populate_h5(f, self.dump)
        else:
            raise ValueError("The model must be saved as a '.json' "
                             f"or '.h5' file, but got '{suffix}'")

    def _populate(self, group: h5py.Group, obj: object):
        """
        Recursively populate an hdf5 file with the object

        Parameters
        ----------
        group : h5py.Group
            an hdf5 group

        h5file
        """
        if isinstance(obj, dict):
            group.attrs["type": "dict"]
            for key, value in obj.items():
                g = group.create_group(key, track_order=True)
                self._populate(g, value)
        elif isinstance(obj, list):
            try:  # Try converting to numpy array
                arr = np.array(obj, dtype=float)
            except ValueError:  # must be saved as a list
                group.attrs["type": "list"]
                for i, value in enumerate(obj):
                    g = group.create_group(f"{i}")
                    self._populate(g, value)
            else:  # save the numpy array as a dataset
                group.attrs["type": "binary"]
                group["data"] = arr
        elif any(isinstance(obj, t) for t in [float, int]):
            pass
        else:
            ValueError(f"Unsupported data type of '{key}': {type(obj)}")
