"""
The seasons - Copernicus
"""
import torch
import pygmalion as ml
import pandas as pd
import matplotlib.pyplot as plt
from pygmalion.datasets.generators import OrbitalTrajectoryGenerator
from pygmalion.neural_networks import TimeSeriesRegressor
from pygmalion.neural_networks.layers.transformers.multihead_attention import FourrierKernelAttention, ScaledDotProductAttention
from pygmalion.neural_networks.layers.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding

DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
model = TimeSeriesRegressor(inputs=[], targets=["x", "y"],
                            observation_column="obj", time_column=None,
                            normalize=False, n_min_points=2,
                            n_stages=1, projection_dim=16, n_heads=4,
                            attention_type=FourrierKernelAttention,
                            attention_kwargs={"linear_complexity": True},
                            positional_encoding_type=LearnedPositionalEncoding,
                            positional_encoding_kwargs={"sequence_length": 1000}
                            )
model.to(DEVICE)


class Batchifyer:

    def __init__(self, n_batches: int, batch_size: int):
        self.data_generator = OrbitalTrajectoryGenerator(n_batches=n_batches, batch_size=batch_size, dt_min=1.0E-4)

    def __iter__(self):
        for batch in self.data_generator:
            yield model.data_to_tensor(self._filter(batch))
    
    def get_batch(self):
        return self._filter(next(iter(self.data_generator)))
    
    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out next steps as soon as data get out of bounds
        """
        filtered = []
        for obj, sub in df.groupby("obj"):
            out_of_bounds = (sub[["x", "y"]].abs() > 3).any(axis=1) | (sub[["u", "v"]].abs() > 10).any(axis=1)
            if out_of_bounds.any():
                i = out_of_bounds.argmax()
                filtered.append(sub.iloc[:i])
            else:
                filtered.append(sub)
        return pd.concat(filtered)


batchifyer = Batchifyer(1, 10)
val_batch = batchifyer.get_batch()
val_data = model.data_to_tensor(val_batch)
train_losses, val_losses, grad, best_step = model.fit(val_data, val_data, n_steps=1_000, keep_best=False)
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step)

val_batch = val_batch[val_batch.obj < 10]
past = val_batch[val_batch.t < 1.0]
future = val_batch[val_batch.t >= 1.0]
df = model.predict(past, future)

f, ax = plt.subplots()
for i, (obj, sub) in enumerate(val_batch.groupby("obj")):
    ax.plot(sub.x, sub.y, color=f"C{i}", linewidth=2)
    ax.scatter(df[df.obj == obj].x, df[df.obj == obj].y, color=f"C{i}", marker=".")

plt.show()
if __name__ == "__main__":
    import IPython
    IPython.embed()