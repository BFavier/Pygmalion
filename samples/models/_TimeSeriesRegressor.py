import torch
import pygmalion as ml
import matplotlib.pyplot as plt
from pygmalion.datasets.generators import OrbitalTrajectoryGenerator
from pygmalion.neural_networks import TimeSeriesRegressor
from pygmalion.neural_networks.layers.transformers import FourrierKernelAttention

DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
model = TimeSeriesRegressor(inputs=["x", "y", "u", "v"],
                            targets=["x", "y", "u", "v"],
                            observation_column="obj", time_column="t",
                            n_stages=4, projection_dim=16, n_heads=4,
                            attention_type=FourrierKernelAttention,
                            attention_kwargs={"linear_complexity": False})
model.to(DEVICE)


class Batchifyer:

    def __init__(self, n_batches: int, batch_size: int):
        self.data_generator = OrbitalTrajectoryGenerator(n_batches=n_batches, batch_size=batch_size, dt_min=1.0E-4, tol=1.0E-6)

    def __iter__(self):
        for batch in self.data_generator:
            yield model.data_to_tensor(batch)
    
    def get_batch(self):
        return next(iter(self.data_generator))


batchifyer = Batchifyer(1, 30)
train_data = next(iter(batchifyer))
train_losses, val_losses, grad, best_step = model.fit(batchifyer, keep_best=False)
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step)
plt.show()

if __name__ == "__main__":
    import IPython
    IPython.embed()