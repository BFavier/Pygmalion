import json
import pathlib


class Model:

    @classmethod
    def load(cls, file: str):
        file = pathlib.Path(file)
        if not file.is_file():
            raise FileNotFoundError("The file '{file}' does not exist")
        with open(file) as json_file:
            data = json.load(json_file)
        return cls.from_dump(data)

    def save(self, file: str, overwrite: bool = False):
        file = pathlib.Path(file)
        path = file.parent
        if not path.is_dir():
            raise ValueError(f"The directory '{path}' does not exist")
        if not(overwrite) and file.exists():
            raise FileExistsError("The file '{file}' already exists,"
                                  " set 'overwrite=True' to overwrite.")
        with open(file, "w") as json_file:
            json.dump(self.dump, json_file)
