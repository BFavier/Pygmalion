from itertools import repeat
import pygmalion as ml
import pandas as pd
import pathlib
import IPython

data_path = pathlib.Path(__file__).parents[1] / "data"
ml.datasets.airline_tweets(data_path)
df = pd.read_csv(data_path / "airline_tweets.csv")


class Batchifyer:

    def __init__(self, df: pd.DataFrame, batch_size: int):
        self.df = df
        self.batch_size = batch_size
    
    def __iter__(self):
        while True:
            yield self.df.sample(n=self.batch_size)



tokenizer = ml.tokenizers.BytePairEncoder(dropout=None, lowercase=True)
count = tokenizer.fit(Batchifyer(df.text, batch_size=100), min_frequency=1.0E-6, pre_tokenize=False)

IPython.embed()
