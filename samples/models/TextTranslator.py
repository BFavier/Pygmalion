import torch
import pygmalion as ml
from pygmalion.tokenizers import AsciiCharTokenizer
from pygmalion.neural_networks import TextTranslator
from pygmalion.neural_networks.layers.positional_encoding import LearnedPositionalEncoding
from pygmalion.datasets.generators import RomanNumeralsGenerator
import IPython
import matplotlib.pyplot as plt

DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
tokenizer = AsciiCharTokenizer()
model = TextTranslator(tokenizer, tokenizer, n_stages=4, projection_dim=16, n_heads=4,
                       positional_encoding_type=LearnedPositionalEncoding,
                       input_positional_encoding_kwargs={"sequence_length": 10},
                       output_positional_encoding_kwargs={"sequence_length": 15})
model.to(DEVICE)


class Batchifyer:
    def __init__(self, model, batch_size: int, n_batches: int=1):
        self.generator = RomanNumeralsGenerator(batch_size, n_batches, max=1999)
        self.model = model
    
    def __iter__(self):
        for arabic_numerals, roman_numerals in self.generator:
            yield self.model.data_to_tensor(arabic_numerals, roman_numerals)

train_data = Batchifyer(model, batch_size=1000)

train_losses, val_losses, grad, best_step = model.fit(train_data, n_steps=3000, learning_rate=1.0E-3)
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step)
plt.show()

for n in torch.randint(0, 1999, size=(10,)):
    print(f"{n} >>> {model.predict(f'{n}')}")
IPython.embed()
