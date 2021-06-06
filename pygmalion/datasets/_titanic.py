from ._download import download


def titanic(directory: str):
    """downloads 'titanic.csv' in the given directory"""
    download(directory, "titanic.csv",
             "https://drive.google.com/file/d/"
             "1LYjbHW3wyJSMzGMMCmaOFNA_RIKqxRoI/view?usp=sharing")
