import torch
import pathlib
import numpy as np
import pygmalion as ml
import pygmalion.neural_networks as nn
import matplotlib.pyplot as plt
from pygmalion.datasets import fashion_mnist
from pygmalion.cross_validation import split
from pygmalion.metrics import confusion_matrix, accuracy
from pygmalion.ploting import plot_matrix, plot_losses
from pygmalion.data_processing import embed_categorical, mask_nullables

plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
fashion_mnist(data_path)

# Load data
data = dict(np.load(data_path / "fashion-MNIST.npz"))
classes, x_train, y_train, x_test, y_test = (data[k] for k in ("classes", "train_images", "train_labels", "test_images", "test_labels"))
classes = [c.decode("utf-8") for c in classes]

# Create and train the model
DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
model = nn.ImageClassifier(1, classes,
                           features=[8, 16, 32],
                           pooling_size=(2, 2),
                           n_convs_per_block=2,
                           activation="relu")
model.to(DEVICE)

class Batchifyer:

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 n_batches: int=1, batch_size: int=1000):
        self.x, self.y = model.data_to_tensor(x, y)
        self.batch_size = batch_size
        self.n_batches = n_batches
    
    def __iter__(self):
        shuffle = torch.randperm(len(self.x))
        for i in range(self.n_batches):
            idx = shuffle[i*self.batch_size:(i+1)*self.batch_size]
            yield (self.x[idx], self.y[idx])

train_data, val_data = (Batchifyer(*data) for data in split(x_train, y_train, weights=(0.8, 0.2)))
train_losses, val_losses, grad, best_step = model.fit(train_data, val_data, n_steps=1000)

# Plot results
plot_losses(train_losses, val_losses, grad, best_step)
f, ax = plt.subplots()
y_pred = sum([model.predict(x_test[i:i+100]) for i in range(0, len(x_test), 100)], [])
y_target = [classes[i] for i in y_test]
plot_matrix(confusion_matrix(y_pred, y_target), ax=ax, color_bar=True)
acc = accuracy(y_pred, y_target)
ax.set_title(f"Accuracy = {acc:.3g}")
f.tight_layout()

# Plot metrics
lx, ly = 5, 5
sample = np.random.permutation(len(x_test))[:lx*ly]
x, y_target = x_test[sample], [y_target[s] for s in sample]
y_pred = model.predict(x)
f, axes = plt.subplots(figsize=[6, 6], ncols=lx, nrows=ly)
for n in range(lx*ly):
    i = n // lx
    j = n % lx
    ax = axes[i, j]
    ax.imshow(x[n], cmap="gray")
    ax.set_title(y_pred[n], color="g" if y_pred[n] == y_target[n] else "r")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
f.tight_layout()

plt.show()
