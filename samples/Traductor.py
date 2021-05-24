import pygmalion as ml
import pathlib
import IPython

data_path = pathlib.Path(__file__).parent / "data"


def load_text(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.read().split("\n")
    return lines


# en = load_text(data_path / "europarl" / "europarl-tokenized-en.txt")[:1000]
# fr = load_text(data_path / "europarl" / "europarl-tokenized-fr.txt")[:1000]
en = ["hello world",
      "My name is Jean",
      "i like trees",
      "i am superman",
      "see you tommorow"]
fr = ["bonjour le monde",
      "je m'appel Jean",
      "j'aime les arbres",
      "je suis superman",
      "on se voit demain"]

tokenizer_in = ml.unsupervised.tokenizers.BytePairEncoder()
tokenizer_in.train(en)
tokenizer_out = ml.unsupervised.tokenizers.WhitespaceTokenizer()
tokenizer_out.train(fr)

n_stages = 4
projection_dim = 16
n_heads = 4
hidden_layers = [{"features": 128}]

model = ml.neural_networks.Traductor(tokenizer_in, tokenizer_out,
                                     n_stages, projection_dim, n_heads,
                                     hidden_layers,
                                     GPU=None, optimization_method="Adam")

model.train((en, fr), n_epochs=101, learning_rate=1.0E-3, batch_size=10)
print(model(en[0], max_words=10))

IPython.embed()
