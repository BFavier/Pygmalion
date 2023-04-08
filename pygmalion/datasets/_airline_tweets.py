import pathlib
from pygmalion.utilities._download import download


def airline_tweets(directory: str):
    """
    Downloads a modified version of the 'Twitter US Airlines Sentiment'
    dataset, in the given directory
    """
    download(pathlib.Path(directory) / "airline_tweets.csv",
             "https://drive.google.com/file/d"
             "/1Lu4iQucxVBncxeyCj_wFKGkq8Wz0-cuL/view?usp=sharing")
