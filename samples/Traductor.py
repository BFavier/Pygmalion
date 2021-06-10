import pygmalion as ml
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import IPython

path = pathlib.Path(__file__).parent
data_path = path / "data"

# Download the data
ml.datasets.sentence_pairs(data_path)

df = pd.read_csv(data_path / "sentence_pairs.txt", header=None,
                 names=["en", "fr"], sep="\t")
en, fr = df["en"].str.lower(), df["fr"].str.lower()

tokenizer_in = ml.unsupervised.tokenizers.WhitespaceTokenizer()
c1 = tokenizer_in.train(en)
tokenizer_out = ml.unsupervised.tokenizers.WhitespaceTokenizer()
c2 = tokenizer_out.train(fr)


n_stages = 2
projection_dim = 16
n_heads = 4
model = ml.neural_networks.Traductor(tokenizer_in, tokenizer_out,
                                     n_stages, projection_dim, n_heads,
                                     GPU=0, optimization_method="Adam")

model.train((en[:500], fr[:500]), n_epochs=100, learning_rate=1.0E-3,
            batch_size=100, n_batches=5)

model.plot_history()
plt.show()

for sentence in en[:3]:
    print()
    print(sentence)
    print(model(sentence, max_words=50))

IPython.embed()
