import pathlib
import json
import torch
import evaluate
from datasets import load_dataset
from tqdm import tqdm
from pygmalion.utilities import plot_losses, load_model
from pygmalion.tokenizers import BytePairEncoder
from pygmalion.neural_networks import TextTranslator
from pygmalion.neural_networks.layers.positional_encoding import SinusoidalPositionalEncoding
from pygmalion.neural_networks.layers.transformers.multihead_attention import FourrierKernelAttention, ScaledDotProductAttention
import numpy as np


method = "vanilla"  # "new" or "vanilla" or "vanilla32"
dataset = load_dataset('wmt14', 'fr-en', trust_remote_code=True)
path = pathlib.Path(__file__).parent
tokenizer_path = path / "tokenizer.json"
restart_step = 0
model_checkpoint_path = path #/ "checkpoints" / "vanilla_model_200000_label_smoothing.pth"
optimizer_checkpoint_path = path #/ "checkpoints" / "optimizer_vanilla_model_200000_label_smoothing.pth"

if tokenizer_path.is_file():
    print("Loading tokenizer")
    tokenizer = load_model(tokenizer_path)
else:
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

if model_checkpoint_path.is_file():
    print("Loading model")
    model = torch.load(model_checkpoint_path)
elif method.startswith("vanilla"):
    model = TextTranslator(tokenizer, tokenizer, n_stages=6,
                           projection_dim=16 if method.endswith("32") else 64,
                           n_heads=32 if method.endswith("32") else 8,
                           dropout=0.1, label_smoothing=0.1,
                           positional_encoding_type=SinusoidalPositionalEncoding,
                           attention_type=ScaledDotProductAttention)
else:
    model = TextTranslator(tokenizer, tokenizer,
                           n_stages=6, projection_dim=16, n_heads=32, dropout=0.1,
                           positional_encoding_type=None, label_smoothing=0.1,
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
            batch = self.dataset.select(np.random.choice(len(self.dataset), size=self.batch_size, replace=False))
            en, fr = zip(*((b["en"], b["fr"]) for b in (b["translation"] for b in batch)))
            yield model.data_to_tensor(en, fr, max_input_sequence_length=self.padded_size,
                                       max_output_sequence_length=self.padded_size)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98))
if optimizer_checkpoint_path.is_file():
    optimizer.load_state_dict(torch.load(optimizer_checkpoint_path, map_location=model.device).state_dict())
hist = model.fit(Batchifyer(dataset["train"], batch_size=100, n_batches=1),
                 Batchifyer(dataset["validation"], batch_size=100, n_batches=1),
                 optimizer, n_steps=300_000, keep_best=False,
                 learning_rate=lambda step: 512**-0.5 * min((step+1)**-0.5, (step+1) * 4000**-1.5),
                 step_offset=restart_step, backup_path=path / "checkpoints", backup_prefix=f"{method}_model", backup_frequency=50_000)

# save model
model.save(path / f"model_{method}.pth", overwrite=True)
# save history
with open(path / f"history_{method}.json", "w", encoding="utf-8") as f:
    json.dump({"train_loss": hist[0], "val_loss": hist[1], "grad": hist[2], "best_step": hist[3]}, f)
# save predictions
with open(path / "checkpoints" / f"{method}_predictions.json", "w") as f:
    for r in tqdm(dataset["test"]):
        r = r["translation"]
        f.write(json.dumps({"predicted": model.predict(r["en"], max_tokens=256), "target": r["fr"]}, ensure_ascii=False)+"\n")
# save BLEU score
bleu = evaluate.load("bleu")
with open(path / "checkpoints" / f"{method}_predictions.json", "r") as h:
    predictions = [json.loads(line) for line in h.read().split("\n") if len(line) > 0]
    with open(path / "checkpoints" / f"{method}_score.json", "w") as f:
        json.dump(bleu.compute(predictions=[p["predicted"][0] for p in predictions], references=[[p["target"]] for p in predictions]), f)

plot_losses(*hist)
import matplotlib.pyplot as plt
plt.show()


if __name__ == "__main__":
    import IPython
    IPython.embed()
