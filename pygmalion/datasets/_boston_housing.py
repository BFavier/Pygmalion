import pathlib
from pygmalion.utilities._download import download


def boston_housing(directory: str):
    """downloads 'boston_housing.csv' in the given directory"""
    download(pathlib.Path(directory) / "boston_housing.csv",
             "https://drive.google.com/file/d"
             "/1fTWYixdKF4tWyhD3V-qCDSZmReN_6LzP/view?usp=sharing")
