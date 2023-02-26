import json
import pathlib
import io
from typing import Union


class ModelBase:

    def __repr__(self):
        return f"{type(self).__name__}()"

    @classmethod
    def load(cls, file_path: Union[str, pathlib.Path, io.IOBase]) -> "ModelBase":
        """
        Load a model from the disk (must be a .json)

        Parameters
        ----------
        file : str or pathlib.Path or file like
            path of the file to read
        """
        if not isinstance(file_path, io.IOBase):
            file_path = pathlib.Path(file_path)
            path = file_path.parent
            suffix = file_path.suffix.lower()
            if suffix != ".json":
                raise ValueError(f"The file must be '.json' file, but got a '{suffix}'")
            if not path.is_dir():
                raise ValueError(f"The directory '{path}' does not exist")
            if not file_path.is_file():
                raise FileNotFoundError(f"The file '{file_path.name}' does not exist in '{path}'")
            with open(file_path) as json_file:
                dump = json.load(json_file)
        else:
            dump = json.load(file_path)
        return cls.from_dump(dump)

    def save(self, file_path: Union[str, pathlib.Path, io.IOBase],
             overwrite: bool = False, create_dir: bool = False):
        """
        Saves the model to the disk as '.json' file

        Parameters
        ----------
        file : str or pathlib.Path or file like
            The path where the file must be created
        overwritte : bool
            If True, the file is overwritten
        create_dir : bool
            If True, the directory to the file's path is created
            if it does not exist already
        """
        if not isinstance(file_path, io.IOBase):
            file_path = pathlib.Path(file_path)
            path = file_path.parent
            suffix = file_path.suffix.lower()
            if suffix != ".json":
                raise ValueError(
                    f"The model must be saved as a '.json' file, but got '{suffix}'")
            if not(create_dir) and not path.is_dir():
                raise ValueError(f"The directory '{path}' does not exist")
            else:
                path.mkdir(exist_ok=True)
            if not(overwrite) and file_path.exists():
                raise FileExistsError(
                    f"The file '{file_path}' already exists, set 'overwrite=True' to overwrite.")
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(self.dump, json_file, ensure_ascii=False)
        else:
            json.dump(self.dump, file_path, ensure_ascii=False)

    @classmethod
    def from_dump(cls, dump: dict) -> "ModelBase":
        raise NotImplementedError()
    
    @property
    def dump(self) -> dict:
        raise NotImplementedError()