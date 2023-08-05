import pathlib
from pygmalion.utilities._download import download


def usa_economy(directory: str):
    """downloads 'usa_economy.csv' in the given directory"""
    download(pathlib.Path(directory) / "usa_economy.csv",
             "https://drive.google.com/file/d/1D9ibXWk6rCRz6RZ8_emFMSnoNlY849VD/view?usp=share_link")
