import torch
import numpy as np
import pygmalion as ml
import matplotlib.pyplot as plt

model = ml.neural_networks.ImageObjectDetector(1, ["circle", "square"], features=[8, 16, 32, 64],
                                               kernel_size=(3, 3), pooling_size=(2, 2), n_convs_per_block=3)


class Batchifyer:

    def __init__(self, batch_size: int, n_batches: int):
        self.generator = ml.datasets.generators.ShapesGenerator(batch_size, n_batches)

    def __iter__(self):
        for x, y in self.generator:
            yield model.data_to_tensor(x, y)


optimizer = torch.optim.Adam(model.parameters())
model.fit(training_data=Batchifyer(100, 1), optimizer=optimizer, n_steps=10000, keep_best=False)

if __name__ == "__main__":
    import IPython
    IPython.embed()
