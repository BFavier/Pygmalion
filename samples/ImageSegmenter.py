import pathlib
import json
import IPython
import numpy as np
import pygmalion as ml
import pygmalion.neural_networks as nn
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / "data"

# Download the data
ml.datasets.cityscapes(data_path)

# Load data
data = dict(np.load(data_path / "cityscapes.npz"))
classes = [c.decode("utf-8") for c in data["classes"].tolist()]
fractions = data["fractions"]
colors = data["colors"]
class_weights = (1/fractions)**0.5
x = data["train_images"][::10]
y = data["train_segmented"][::10]
x_test = data["test_images"][::10]
y_test = data["test_segmented"][::10]

# Create and train the model
device = "cuda:0"
model = nn.ImageSegmenter(3, classes, [8, 16, 32, 64], pooling_size=(2, 2), stride=(2, 2), n_convs_per_block=2)
model.to(device)

class Batchifyer:

    def __init__(self, x: np.ndarray, y: np.ndarray, class_weights: np.ndarray,
                 n_batches: int=1, batch_size: int=10, device: str=device):
        self.x, self.y, self.class_weights = x, y, class_weights
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.device = device
    
    def __iter__(self):
        shuffle = np.random.permutation(len(self.x))
        for i in range(self.n_batches):
            idx = shuffle[i*self.batch_size:(i+1)*self.batch_size]
            yield model.data_to_tensor(self.x[idx], self.y[idx],
                                       class_weights=self.class_weights,
                                       device=self.device)

train_split, val_split = ml.split(x, y, weights=(0.8, 0.2))
train_data, val_data = Batchifyer(*train_split, class_weights), Batchifyer(*val_split, class_weights)
train_losses, val_losses, best_step = model.fit(train_data, val_data,
    n_steps=5000, learning_rate=1.0E-4, patience=100)

# Plot results
ml.plot_losses(train_losses, val_losses, best_step)
x_train, y_train = train_split
x_train, y_train = x_train[:5], y_train[:5]
y_predicted = model.predict(x_train)
for x, y_t, y_p in zip(x_train, y_train, y_predicted):
    f, axes = plt.subplots(figsize=[15, 5], ncols=3)
    for im, ax, title in zip([x, y_p, y_t], axes,
                             ["image", "segmented", "target"]):
        if len(im.shape) == 2:
            h, w = im.shape
            n, c = colors.shape
            im = np.take_along_axis(colors.reshape(1, 1, n, c),
                                    im.reshape(h, w, 1, 1), axis=-2).reshape(h, w, c)
        ax.imshow(im)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    f.tight_layout()
plt.show()

IPython.embed()
