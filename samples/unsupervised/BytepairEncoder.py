import pygmalion as ml
import pandas as pd
import pathlib
import IPython

data_path = pathlib.Path(__file__).parents[1] / "data"
ml.datasets.airline_tweets(data_path)
df = pd.read_csv(data_path / "airline_tweets.csv")
tokenizer = ml.unsupervised.tokenizers.BytePairEncoder(dropout=0.1)
count = tokenizer.train(df.text.str.lower(), min_frequency=1.0E-6)

IPython.embed()