import pathlib
from pygmalion.utilities._download import download


def iris(directory: str):
    """downloads 'iris.csv' in the given directory"""
    download(pathlib.Path(directory) / "iris.csv",
             "https://drive.google.com/file/d/1S1AHfTBtnW1SxsMskRmUcnoDehMbCj0R/view?usp=sharing")
