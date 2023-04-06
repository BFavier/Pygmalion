from typing import Iterable
import pygmalion as ml
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import IPython
import torch

path = pathlib.Path(__file__).parent
data_path = path.parent / "data"

# Download the data
ml.datasets.sentence_pairs(data_path)

df = pd.read_csv(data_path / "sentence_pairs.csv.gz")
df_train = df.sample(frac=0.9)
df_val = df.drop(index=df_train.index)


class Looper:

    def __init__(self, series: Iterable[str], batch_size: int):
        self.series = pd.Series(series)
        self.batch_size = batch_size
    
    def __iter__(self):
        while True:
            yield self.series.sample(n=self.batch_size)

tok_in_file = path / "tokenizer_in.json"
if tok_in_file.is_file():
    tokenizer_in = ml.tokenizers.BytePairEncoder.load(tok_in_file)
else:
    tokenizer_in = ml.tokenizers.BytePairEncoder()
    tokenizer_in.fit(Looper(df.fr, 1000), max_vocabulary_size=10000)
    tokenizer_in.save(tok_in_file)
tok_out_file = path / "tokenizer_out.json"
if tok_out_file.is_file():
    tokenizer_out = ml.tokenizers.BytePairEncoder.load(tok_out_file)
else:
    tokenizer_out = ml.tokenizers.BytePairEncoder()
    tokenizer_out.fit(Looper(df.en, 1000), max_vocabulary_size=10000)
    tokenizer_out.save(tok_out_file)

model = ml.neural_networks.TextTranslator(tokenizer_in, tokenizer_out, n_stages=6, projection_dim=64, n_heads=8,
                                          RPE_radius=8, dropout=0.1,
                                          positional_encoding_type=None)
model.to("cuda:0")

class Batchifyer:

    def __init__(self, df: pd.DataFrame, model: ml.neural_networks.TextTranslator,
                 batch_size: int, n_batches: int, sequence_length: int = 128):
        self.x, self.y = model.data_to_tensor(df.fr, df.en,
                                              max_input_sequence_length=sequence_length,
                                              max_output_sequence_length=sequence_length,
                                              progress_bar=True)
        # self.df = df
        # self.model = model
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            indexes = torch.randperm(len(self.x))[:self.batch_size]
            yield (self.x[indexes], self.y[indexes])

    # def __iter__(self):
    #     for _ in range(self.n_batches):
    #         sub = self.df.sample(n=self.batch_size)
    #         yield self.model.data_to_tensor(sub.fr, sub.en,
    #                                         max_input_sequence_length=128,
    #                                         max_output_sequence_length=128)

train = Batchifyer(df_train, model, batch_size=800, n_batches=1)
val = Batchifyer(df_val, model, batch_size=800, n_batches=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0., betas=(0.9, 0.98))
train_losses, val_losses, grad, best_step = model.fit(train, val, optimizer,
    n_steps=100000, patience=None, keep_best=True,
    learning_rate=lambda step: 512**-0.5 * min((step+1)**-0.5, (step+1) * 4000**-1.5))

ml.plot_losses(train_losses, val_losses, grad, best_step);plt.show()
IPython.embed()