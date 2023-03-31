import pathlib
from ._download import download

def cityscapes(directory: str):
    """downloads the modified 'cityscapes' dataset in the given directory"""
    download(pathlib.Path(directory) / "cityscapes.npz",
             "https://drive.google.com/file/d/1PtB9qGFpf2c1ltqSuzYgTHOKjjI3uDIW/view?usp=share_link")
