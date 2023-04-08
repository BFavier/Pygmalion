import json
import pathlib
import io
from typing import Union


class Model:

    def __repr__(self):
        return f"{type(self).__name__}()"

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
    def from_dump(cls, dump: dict) -> "Model":
        raise NotImplementedError()
    
    @property
    def dump(self) -> dict:
        raise NotImplementedError()