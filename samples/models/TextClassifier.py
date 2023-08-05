import pygmalion as ml
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pathlib
import IPython
from pygmalion.neural_networks.layers.positional_encoding import LearnedPositionalEncoding
from pygmalion.neural_networks.layers.transformers.multihead_attention import KernelizedAttention

path = pathlib.Path(__file__).parent
data_path = path.parent / "data"

# Download the data
ml.datasets.airline_tweets(data_path)
df = pd.read_csv(data_path / "airline_tweets.csv")
class_freqs = pd.value_counts(df["sentiment"], normalize=True)
classes = class_freqs.index
class_weights = {c: f**-0.5 for c, f in class_freqs.items()}

tokenizer = ml.tokenizers.WordsTokenizer(lowercase=True, special_tokens=["UNKNOWN", "PAD"])
tokenizer.fit(df["text"], max_tokens=10000)
model = ml.neural_networks.TextClassifier(classes, tokenizer,
                                          n_stages=3, projection_dim=16,
                                          n_heads=4, dropout=0.2,
                                          positional_encoding_type=LearnedPositionalEncoding,
                                          positional_encoding_kwargs={"sequence_length": 100},
                                          attention_type=KernelizedAttention,
                                          attention_kwargs={"linear_complexity": False})
DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
model.to(DEVICE)


class Batchifyer:

    def __init__(self, df: pd.DataFrame, batch_size: int):
        self.x, self.y, w, self.cw = model.data_to_tensor(df["text"], df["sentiment"], class_weights=[class_weights[c] for c in classes])
        self.batch_size = batch_size

    def __iter__(self):
        index = torch.randperm(len(self.x))[:self.batch_size]
        yield (self.x[index], self.y[index], None, self.cw)


df_train, df_val = ml.utilities.split(df)
train_data, val_data = Batchifyer(df_train, 1000), Batchifyer(df_val, 1000)
train_loss, val_loss, grad, best_step = model.fit(train_data, validation_data=val_data, n_steps=1000, learning_rate=1.0E-3)
ml.utilities.plot_losses(train_loss, val_loss, grad, best_step)

y_val = df_val["sentiment"]
y_pred = model.predict(df_val["text"])
f, ax = plt.subplots()
ml.utilities.plot_matrix(ml.utilities.confusion_matrix(y_val, y_pred), ax=ax, write_values=True,
               format=".2%", cmap="Greens")
ax.set_xlabel("target")
ax.set_ylabel("predicted")
plt.show()

IPython.embed()
