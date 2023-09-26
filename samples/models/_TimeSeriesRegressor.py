import torch
import pygmalion as ml
import matplotlib.pyplot as plt
from pygmalion.datasets.generators import OrbitalTrajectoryGenerator
from pygmalion.neural_networks import TimeSeriesRegressor
from pygmalion.neural_networks.layers.transformers.multihead_attention import FourrierKernelAttention

DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
model = TimeSeriesRegressor(inputs=["x", "y"], targets=["x", "y"],
                            observation_column="obj", time_column="t",
                            normalize=True, std_noise=0., n_min_points=2,
                            n_stages=6, projection_dim=16, n_heads=8,
                            attention_type=FourrierKernelAttention,
                            attention_kwargs={"linear_complexity": True}
                            )
model.to(DEVICE)


class Batchifyer:

    def __init__(self, n_batches: int, batch_size: int):
        self.data_generator = OrbitalTrajectoryGenerator(n_batches=n_batches, batch_size=batch_size, dt_min=1.0E-6, tol=1.0E-6)

    def __iter__(self):
        for batch in self.data_generator:
            yield model.data_to_tensor(batch)
    
    def get_batch(self):
        return next(iter(self.data_generator))


batchifyer = Batchifyer(3, 10)
val_batch = batchifyer.get_batch()
val_data = model.data_to_tensor(val_batch)
train_losses, val_losses, grad, best_step = model.fit(val_data, val_data, keep_best=False, learning_rate=1.0E-5)
model.cpu()
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step)

past = val_batch[val_batch.t < 0.5]
future = val_batch[val_batch.t >= 0.5]
df = model.predict(past, future)

f, ax = plt.subplots()
for i, (obj, sub) in enumerate(val_batch.groupby("obj")):
    ax.plot(sub.x, sub.y, color=f"C{i}", linewidth=2)
    ax.scatter(df[df.obj == obj].x, df[df.obj == obj].y, color=f"C{i}", marker=".")
plt.show()
if __name__ == "__main__":
    import IPython
    IPython.embed()