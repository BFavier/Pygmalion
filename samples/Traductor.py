import pygmalion as ml
import pandas as pd
import pathlib
import IPython
import matplotlib.pyplot as plt
import numpy as np
import itertools
import spacy

data_path = pathlib.Path(__file__).parent / "data"


def load_text(path):
    with open(path, "r", encoding="utf-8") as file:
        raw = [line.encode("utf-8") for line in file]
    Lmax = max(len(line) for line in raw)
    padded = [b"\r" + line + b"\n" * (Lmax - len(line)) for line in raw]
    longs = np.array([list(p) for p in padded])
    return longs


# en = load_text(data_path / "europarl" / "europarl-tokenized-en.txt")
# fr = load_text(data_path / "europarl" / "europarl-tokenized-fr.txt")

en = ["hello world"]
fr = ["bonjour le monde"]
lexicon_in, lexicon_out = en[0].split(), fr[0].split()


projection_dim = 16
n_heads = 4
layers = [{"channels": 128}]
n_stages = 4

model = ml.neural_networks.Traductor(embedding_dim, lexicon_in, lexicon_out,
                                     projection_dim, n_heads, layers, n_stages,
                                     GPU=0)

model.train((en, fr), n_epochs=500, learning_rate=1.0E-3)
print(model(en[0], max_words=10).split(" "))

IPython.embed()
