import pygmalion as ml
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import IPython

path = pathlib.Path(__file__).parent
data_path = path / "data"

# Download the data
ml.datasets.sentence_pairs(data_path)


# def load_text(path):
#     with open(path, "r", encoding="utf-8") as file:
#         lines = file.read().split("\n")
#     return lines


# en = load_text(data_path / "europarl" / "europarl-tokenized-en.txt")
# fr = load_text(data_path / "europarl" / "europarl-tokenized-fr.txt")
df = pd.read_csv(data_path / "sentence_pairs.txt", header=None,
                 names=["en", "fr"], sep="\t")

en, fr = df["en"].str.lower(), df["fr"].str.lower()
# en = ["hello world",
#       "My name is Jean",
#       "i like trees",
#       "i am superman",
#       "see you tommorow"]
# fr = ["bonjour le monde",
#       "je m'appel Jean",
#       "j'aime les arbres",
#       "je suis superman",
#       "on se voit demain"]

import pickle
with open(path / "tokenizer_in.pk", "rb") as file:
    tokenizer_in = pickle.load(file)
with open(path / "tokenizer_out.pk", "rb") as file:
    tokenizer_out = pickle.load(file)
# tokenizer_in = ml.unsupervised.tokenizers.BytePairEncoder()
# c1 = tokenizer_in.train(en, min_frequency=1.0E-5)
# tokenizer_out = ml.unsupervised.tokenizers.BytePairEncoder()
# c2 = tokenizer_out.train(fr, min_frequency=1.0E-5)


# IPython.embed()

n_stages = 2
projection_dim = 16
n_heads = 4
hidden_layers = [{"features": 128}]

model = ml.neural_networks.Traductor(tokenizer_in, tokenizer_out,
                                     n_stages, projection_dim, n_heads,
                                     hidden_layers,
                                     GPU=0, optimization_method="Adam")

model.train((en[:500], fr[:500]), n_epochs=1000, learning_rate=1.0E-3)

model.plot_history()
plt.show()

for sentence in en[:3]:
    print()
    print(sentence)
    print(model(sentence, max_words=50))

IPython.embed()
