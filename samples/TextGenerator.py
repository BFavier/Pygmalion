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

x = df["text"].str.lower()

tokenizer = ml.unsupervised.tokenizers.WhitespaceTokenizer()
c = tokenizer.train(x, max_tokens=1000)

n_stages = 3
projection_dim = 16
n_heads = 4

model = ml.neural_networks.TextGenerator(tokenizer, n_stages, projection_dim,
                                         n_heads, GPU=0,
                                         optimization_method="Adam")
train_data, val_data = ml.split(x, frac=0.2)
model.train(train_data, validation_data=val_data, n_epochs=1000,
            learning_rate=1.0E-3, batch_size=100, n_batches=5)
model.plot_history()
plt.show()

print(model(p=0.))

IPython.embed()
