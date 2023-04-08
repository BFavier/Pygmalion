import pathlib
from ._download import download


def aquarium(directory: str):
    """downloads the 'fashion MNIST' dataset in the given directory"""
    download(pathlib.Path(directory) / "aquarium.npz",
             "https://drive.google.com/file/d/1ZiVbN5D0JXyAvha2NcyG3q-NF2cl6fO_/view?usp=share_link")
