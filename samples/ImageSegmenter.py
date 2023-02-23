import pathlib
import json
import IPython
import numpy as np
import pygmalion as ml
import pygmalion.neural_networks as nn
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / "data" / "cityscapes"

# Download the data
ml.datasets.cityscapes(data_path.parent)

# Load data
with open(data_path / "class_fractions.json", "r") as file:
    fractions = json.load(file)
class_weights = {k: (1/f)**0.5 if f > 0 else 0. for k, f in fractions.items()}
with open(data_path / "classes.json", "r") as file:
    classes = json.load(file)
x = np.load(data_path / "train_images.npy")[::10]
y = np.load(data_path / "train_segmented.npy")[::10]
x_test = np.load(data_path / "test_images.npy")[::10]
y_test = np.load(data_path / "test_segmented.npy")[::10]

# Create and train the model
device = "cuda:0"
model = nn.ImageSegmenter(3, classes, [8, 16, 32, 64, 128], pooling_size=(2, 2), n_convs_per_block=2)
model.to(device)

class Batchifyer:

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 n_batches: int=1, batch_size: int=10, device: str=device):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.device = device
    
    def __iter__(self):
        shuffle = np.random.permutation(len(self.x))
        for i in range(self.n_batches):
            idx = shuffle[i*self.batch_size:(i+1)*self.batch_size]
            yield model.data_to_tensor(self.x[idx], self.y[idx],
                                       class_weights=class_weights,
                                       device=self.device)

train_split, val_split = ml.split(x, y, weights=(0.8, 0.2))
train_data, val_data = Batchifyer(*train_split), Batchifyer(*val_split)
train_losses, val_losses, best_step = model.fit(train_data, val_data,
    n_steps=5000, learning_rate=1.0E-3, patience=100)

# Plot results
ml.plot_losses(train_losses, val_losses, best_step)
x_train, y_train = train_split
for x, y_t in zip(x_train[:5], y_train[:5]):
    y_p = model([x])[0]
    f, axes = plt.subplots(figsize=[15, 5], ncols=3)
    for im, ax, title in zip([x, y_p, y_t], axes,
                             ["image", "segmented", "target"]):
        ax.imshow(im)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    f.tight_layout()
plt.show()

IPython.embed()
