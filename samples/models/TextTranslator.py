import pygmalion as ml
import IPython
import matplotlib.pyplot as plt

DEVICE = "cuda:0"
tokenizer = ml.tokenizers.AsciiCharTokenizer()
model = ml.neural_networks.TextTranslator(tokenizer, tokenizer, n_stages=4, projection_dim=16, n_heads=4)
model.to(DEVICE)

class Batchifyer:
    def __init__(self, model, batch_size: int, n_batches: int=1):
        self.generator = ml.datasets.generators.RomanNumeralsGenerator(batch_size, n_batches, max=1999)
        self.model = model
    
    def __iter__(self):
        for arabic_numerals, roman_numerals in self.generator:
            yield self.model.data_to_tensor(arabic_numerals, roman_numerals)

train_data = Batchifyer(model, batch_size=10000)

train_losses, val_losses, best_step = model.fit(train_data, n_steps=1000, learning_rate=1.0E-3)

ml.plot_losses(train_losses, val_losses, best_step)
plt.show()
IPython.embed()
