import pathlib
import json
import torch
from datasets import load_dataset
from pygmalion.utilities import plot_losses, load_model
from pygmalion.tokenizers import BytePairEncoder
from pygmalion.neural_networks import TextTranslator
from pygmalion.neural_networks.layers.positional_encoding import LearnedPositionalEncoding
from pygmalion.neural_networks.layers.transformers.multihead_attention import FourrierKernelAttention, ScaledDotProductAttention
import pandas as pd


method = "new"  # "new" or "vanilla"
dataset = load_dataset('wmt14', 'fr-en', trust_remote_code=True)
path = pathlib.Path(__file__).parent
tokenizer_path = path / "tokenizer.json"
if not tokenizer_path.is_file():
    tokenizer = BytePairEncoder()

    
    class StringsBatchifyer:

        def __init__(self, data: object):
            self.generator = (string for obs in data for string in (obs["translation"]["fr"], obs["translation"]["en"]))
        
        def __iter__(self):
            while True:
                yield [string for _, string in zip(range(100), self.generator)]

    tokenizer.fit(StringsBatchifyer(dataset["train"]),
                  max_vocabulary_size=20_000, min_frequency=0, pre_tokenize=True)
    tokenizer.save(tokenizer_path)
else:
    tokenizer = load_model(tokenizer_path)

if method == "vanilla":
    model = TextTranslator(tokenizer, tokenizer,
                           n_stages=6, projection_dim=64, n_heads=8, dropout=None,
                           positional_encoding_type=LearnedPositionalEncoding,
                           input_positional_encoding_kwargs={"sequence_length": 256},
                           output_positional_encoding_kwargs={"sequence_length": 256},
                           attention_type=ScaledDotProductAttention)
else:
    model = TextTranslator(tokenizer, tokenizer,
                           n_stages=6, projection_dim=16, n_heads=32, dropout=None,
                           positional_encoding_type=None,
                           attention_type=FourrierKernelAttention)
print(method, f"{sum(p.numel() for p in model.parameters()):.3g} parameters")
model.to("cuda:0")


class Batchifyer:

    def __init__(self, dataset: object, batch_size: int=100, n_batches: int=10, padded_size:int=256):
        self.dataset = dataset
        self.iterator = iter(self.dataset)
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.padded_size = padded_size
    
    def __iter__(self):
        for _ in range(self.n_batches):
            fr, en = [], []
            for _ in range(self.batch_size):
                row = next(self.iterator, None)
                if row is None:
                    self.iterator = iter(self.dataset)
                    row = next(self.iterator)
                trsl = row["translation"]
                fr.append(trsl["fr"])
                en.append(trsl["en"])
            yield model.data_to_tensor(fr, en, max_input_sequence_length=self.padded_size,
                                       max_output_sequence_length=self.padded_size)


optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98))
hist = model.fit(Batchifyer(dataset["train"], batch_size=200, n_batches=1),
                 Batchifyer(dataset["validation"], batch_size=200, n_batches=1),
                 optimizer, n_steps=30_000, keep_best=False,
                 learning_rate=lambda step: 1.0E-3 * 10**-(step/15_000),
                 backup_path=path / "checkpoints", backup_prefix=f"{method}_model_", backup_frequency=5_000)
model.save(path / f"model_{method}.pth", overwrite=True)
with open(path / f"history_{method}.json", "w", encoding="utf-8") as f:
    json.dump({"train_loss": hist[0], "val_loss": hist[1], "grad": hist[2], "best_step": hist[3]}, f)
plot_losses(*hist)
import matplotlib.pyplot as plt
plt.show()


if __name__ == "__main__":
    import IPython
    IPython.embed()
