from ._download import download


def airline_tweets(directory: str):
    """downloads 'boston_housing.csv' in the given directory"""
    download(directory, "airline_tweets.csv", "https://drive.google.com/file/d/1Lu4iQucxVBncxeyCj_wFKGkq8Wz0-cuL/view?usp=sharing")
