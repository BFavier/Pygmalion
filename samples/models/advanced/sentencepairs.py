from typing import Iterable
import pygmalion as ml
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import IPython
import torch

path = pathlib.Path(__file__).parents[2]
data_path = path / "data"

# Download the data
# ml.datasets.sentence_pairs(data_path)

df = pd.read_csv(data_path / "QED_v2.0a.csv.gz")
df_train = df.sample(frac=0.8)
df_val = df.drop(index=df_train.index)

class Looper:

    def __init__(self, series: Iterable[str], batch_size: int):
        self.series = pd.Series(series)
        self.batch_size = batch_size
    
    def __iter__(self):
        while True:
            yield self.series.sample(n=self.batch_size)

tokenizer_in = ml.tokenizers.BytePairEncoder()
tokenizer_in.fit(Looper(df.fr, 1000), max_vocabulary_size=20000)
tokenizer_out = ml.tokenizers.BytePairEncoder()
tokenizer_out.fit(Looper(df.en, 1000), max_vocabulary_size=20000)

model = ml.neural_networks.TextTranslator(tokenizer_in, tokenizer_out, n_stages=6, projection_dim=16, n_heads=8,
                                          positional_encoding_type=None, RPE_radius=16, dropout=0.1)
model.to("cuda:0")

class Batchifyer:

    def __init__(self, df: pd.DataFrame, model: ml.neural_networks.TextTranslator, batch_size: int):
        self.df = df
        self.batch_size = batch_size
    
    def __iter__(self):
        sub = df.sample(n=self.batch_size)
        yield model.data_to_tensor(sub.fr, sub.en)

train = Batchifyer(df_train, model, batch_size=100)
val = Batchifyer(df_val, model, batch_size=100)

train_losses, val_losses, best_step = model.fit(train, val, n_steps=3000, learning_rate=1.0E-3)
ml.plot_losses(train_losses, val_losses, best_step)
plt.show()
IPython.embed()