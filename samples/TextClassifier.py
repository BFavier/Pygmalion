import pygmalion as ml
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import IPython

path = pathlib.Path(__file__).parent
data_path = path / "data"

# Download the data
ml.datasets.airline_tweets(data_path)
df = pd.read_csv(data_path / "airline_tweets.csv")

x, y = df["text"].str.lower(), df["sentiment"]
class_counts = pd.value_counts(y)
classes = class_counts.index
class_weights = {c: 1/n for c, n in class_counts.items()}
s = sum(class_weights.values())
class_weights = {c: v/s for c, v in class_weights.items()}

tokenizer = ml.unsupervised.tokenizers.WhitespaceTokenizer()
c = tokenizer.train(x, min_frequency=1.0E-5)

n_stages = 3
projection_dim = 16
n_heads = 4

model = ml.neural_networks.TextClassifier(tokenizer, classes,
                                          n_stages, projection_dim,
                                          n_heads, dropout=0.2, GPU=0,
                                          optimization_method="Adam")
train_data, val_data = ml.split(x, y, frac=0.2)
model.train(train_data, validation_data=val_data, n_epochs=1000,
            learning_rate=1.0E-3, batch_size=500, n_batches=5)
model.plot_history()

x_val, y_val = val_data
y_pred = model(x_val, batch_size=500)
f, ax = plt.subplots()
ml.plot_matrix(ml.confusion_matrix(y_val, y_pred), ax=ax, write_values=True,
               format=".2%", cmap="Greens")
ax.set_xlabel("target")
ax.set_ylabel("predicted")
plt.show()

IPython.embed()
