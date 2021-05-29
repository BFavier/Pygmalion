from ._download import download


def sentence_pairs(directory: str):
    """downloads 'boston_housing.csv' in the given directory"""
    download(directory, "sentence_pairs.txt", "https://drive.google.com/file/d/12IiCD6Gm4kFHSO2GtXPipgsjdW89e338/view?usp=sharing")