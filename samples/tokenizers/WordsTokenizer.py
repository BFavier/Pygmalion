import pygmalion as ml
import pandas as pd
import pathlib
import IPython

data_path = pathlib.Path(__file__).parents[1] / "data"
ml.datasets.airline_tweets(data_path)
df = pd.read_csv(data_path / "airline_tweets.csv")
tokenizer = ml.tokenizers.WordsTokenizer(lowercase=True, ascii=False)
count = tokenizer.fit(df.text, max_tokens=5000)

IPython.embed()
