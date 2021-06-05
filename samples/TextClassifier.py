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
classes = y.unique()

tokenizer = ml.unsupervised.tokenizers.WhitespaceTokenizer()
c = tokenizer.train(x, min_frequency=1.0E-5)

n_stages = 2
projection_dim = 16
n_heads = 4
hidden_layers = [{"features": 128}]

model = ml.neural_networks.TextClassifier(tokenizer, classes,
                                          n_stages, projection_dim,
                                          n_heads, hidden_layers, GPU=0,
                                          optimization_method="Adam")

print("training ...", flush=True)

model.train((x, y), n_epochs=1000, learning_rate=1.0E-3, batch_size=1000)
model.plot_history()


print("evaluating ...")
y_pred = model(x, batch_size=500)
ml.plot_matrix(ml.confusion_matrix(y, y_pred))
plt.show()

IPython.embed()
