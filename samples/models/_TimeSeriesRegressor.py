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
from pygmalion.neural_networks.layers.transformers.multihead_attention import FourrierKernelAttention, ScaledDotProductAttention

DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
model = TimeSeriesRegressor(inputs=["x", "y"], targets=["x", "y"], observation_column="obj",
                            time_column="t", normalize=False,
                            n_stages=4, projection_dim=16, n_heads=4)
model.to(DEVICE)


class Batchifyer:

    def __init__(self, n_batches: int, batch_size: int,
                 device: torch.device = "cpu", sequence_length: int=501):
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.data_generator = OrbitalTrajectoryGenerator(n_batches=n_batches, batch_size=batch_size,
                                                         T=np.linspace(0.0, 5.0, sequence_length), dt_min=1.0E-4)

    def __iter__(self):
        for batch in self.data_generator:
            yield model.data_to_tensor(self._filter(batch), self.device,
                                       self.sequence_length-1, self.sequence_length-2)

    def get_batch(self):
        return self._filter(next(iter(self.data_generator)))
    
    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out next steps as soon as data get out of bounds
        """
        inputs = []
        targets = []
        for obj, sub in df.groupby("obj"):
            out_of_bounds = (sub[["x", "y"]].abs() > 3).any(axis=1)  # | (sub[["u", "v"]].abs() > 10).any(axis=1)
            if out_of_bounds.any():
                i = out_of_bounds.argmax()
                sub = sub.iloc[:i]
            i = random.randint(2, len(sub)-1)
            inputs.append(sub.iloc[:i])
            targets.append(sub.iloc[i:])
        return pd.concat(inputs), pd.concat(targets)


batchifyer = Batchifyer(1, 10)
val_batch = batchifyer.get_batch()
val_data = model.data_to_tensor(*val_batch, model.device)
train_losses, val_losses, grad, best_step = model.fit(val_data, val_data, n_steps=1_000, keep_best=False, learning_rate=1.0E-4)
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step)

past, future = val_batch
df = model.predict(past, future.t)

f, ax = plt.subplots()
for i, (obj, sub) in enumerate(past.groupby("obj")):
    ax.plot(list(sub.x), list(sub.y), color=f"C{i}", linewidth=2, label=obj)
    fut = future[future.obj == obj]
    ax.plot(list(fut.x), list(fut.y), color=f"C{i}", linewidth=0.5)
    ax.scatter(list(df[df.obj == obj].x), list(df[df.obj == obj].y), color=f"C{i}", marker=".")
plt.show()

if __name__ == "__main__":
    import IPython
    IPython.embed()