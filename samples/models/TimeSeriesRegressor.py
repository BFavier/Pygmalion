"""
The seasons - Copernicus
"""
import random
import torch
import pygmalion as ml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygmalion.datasets.generators import OrbitalTrajectoryGenerator
from pygmalion.neural_networks import TimeSeriesRegressor
from pygmalion.neural_networks.layers.transformers.multihead_attention import FourrierKernelAttention

DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
model = TimeSeriesRegressor(inputs=["x", "y"], targets=["x", "y"],
                            observation_column="ID", time_column="t",
                            n_stages=4, projection_dim=16, n_heads=4,
                            positional_encoding=False,
                            gradient_checkpointing=False,
                            attention_type=FourrierKernelAttention,
                            send_time_to_attention=True,
                            attention_kwargs={"linear_complexity": True}
                            )
model.to(DEVICE)


class Batchifyer:

    def __init__(self, n_batches: int, batch_size: int,
                 device: torch.device = "cpu", sequence_length: int=101):
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.data_generator = OrbitalTrajectoryGenerator(n_batches=n_batches, batch_size=batch_size,
                                                         n_steps=sequence_length)

    def __iter__(self):
        for batch in self.data_generator:
            yield model.data_to_tensor(*self._split(batch), self.device,
                                       self.sequence_length-1, self.sequence_length-2)

    def get_batch(self):
        return self._split(next(iter(self.data_generator)))
    
    def _split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split the input and target time steps for each orbital trajectory
        """
        inputs = []
        targets = []
        for ID, sub in df.groupby("ID"):
            i = random.randint(10, len(sub)-10)
            inputs.append(sub.iloc[:i])
            targets.append(sub.iloc[i:])
        return pd.concat(inputs), pd.concat(targets)


train_data = Batchifyer(1, 1_000)
train_losses, val_losses, grad, best_step = model.fit(train_data, n_steps=1_000, keep_best=False, learning_rate=1.0E-3)
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step)

# testing trained model
past, future = Batchifyer(1, 10).get_batch()
df = model.predict(past, future[["ID", "t"]])
f, ax = plt.subplots()
for i, (ID, sub) in enumerate(past.groupby("ID")):
    full = pd.concat([sub, future[future.ID == ID]])
    ax.plot(list(full.x), list(full.y), color=f"C{i}", linewidth=1.0, linestyle="--")
    ax.plot(list(sub.x), list(sub.y), color=f"C{i}", linewidth=1.0, label=ID)
    ax.scatter(list(df[df.ID == ID].x), list(df[df.ID == ID].y), color=f"C{i}", marker=".")
ax.set_xticks([])
ax.set_yticks([])
f.tight_layout()
plt.show()

if __name__ == "__main__":
    import IPython
    IPython.embed()