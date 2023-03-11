import pygmalion as ml
import torch
import IPython

DEVICE = "cpu"#"cuda:0"
tokenizer = ml.tokenizers.AsciiCharTokenizer()
model = ml.neural_networks.TextTranslator(tokenizer, tokenizer, n_stages=4, projection_dim=16, n_heads=4)
model.to(DEVICE)

class Batchifyer:
    def __init__(self, tokenizer: ml.tokenizers.AsciiCharTokenizer, batch_size: int, n_batches: int=1):
        self.generator = ml.datasets.generators.RomanNumeralsGenerator(batch_size, n_batches)
        self.tokenizer = tokenizer
    
    def __iter__(self):
        for arabic_numerals, roman_numerals in self.generator:
            yield model.data_to_tensor(arabic_numerals, roman_numerals)

train_data = Batchifyer(tokenizer, batch_size=100)

train_losses, val_losses, best_step = model.fit(train_data, n_steps=1000, learning_rate=1.0E-3)

IPython.embed()
