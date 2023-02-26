import pathlib
import IPython
import numpy as np
import pygmalion as ml
import pygmalion.neural_networks as nn
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data" / "fashion-MNIST"

# Download the data
ml.datasets.fashion_mnist(data_path.parent)

# Load data
with open(data_path / "classes.txt", "r") as file:
    data = file.read()
classes = data.split(",")
x_train = np.load(data_path / "train_images.npy")
y_train = np.array(classes)[np.load(data_path / "train_labels.npy")]
x_test = np.load(data_path / "test_images.npy")
y_test = np.array(classes)[np.load(data_path / "test_labels.npy")]

# Create and train the model
device = "cuda:0"
model = nn.ImageClassifier(1, classes,
                           features=[8, 16, 32],
                           pooling_size=(2, 2),
                           n_convs_per_block=2,
                           activation="relu")
model.to(device)

class Batchifyer:

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 n_batches: int=1, batch_size: int=1000, device: str=device):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.device = device
    
    def __iter__(self):
        shuffle = np.random.permutation(len(self.x))
        for i in range(self.n_batches):
            idx = shuffle[i*self.batch_size:(i+1)*self.batch_size]
            yield model.data_to_tensor(self.x[idx], self.y[idx], device=self.device)

train_data, val_data = (Batchifyer(*data) for data in ml.split(x_train, y_train, weights=(0.8, 0.2)))
train_losses, val_losses, best_step = model.fit(train_data, val_data, n_steps=300)

# Plot results
ml.plot_losses(train_losses, val_losses, best_step)
f, ax = plt.subplots()
y_pred = sum([model.predict(x_test[i:i+100]) for i in range(0, len(x_test), 100)], [])
ml.plot_matrix(ml.confusion_matrix(y_pred, y_test), ax=ax, color_bar=True)
acc = ml.accuracy(y_pred, y_test)
ax.set_title(f"Accuracy = {acc:.3g}")
f.tight_layout()

# Plot metrics
lx, ly = 5, 5
sample = np.random.permutation(len(x_test))[:lx*ly]
x, y_target = x_test[sample], y_test[sample]
y_pred = model.predict(x)
f, axes = plt.subplots(figsize=[6, 6], ncols=lx, nrows=ly)
for n in range(lx*ly):
    i = n // lx
    j = n % lx
    ax = axes[i, j]
    ax.imshow(x[n], cmap="gray_r")
    ax.set_title(y_pred[n], color="g" if y_pred[n] == y_target[n] else "r")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
f.tight_layout()

plt.show()
IPython.embed()
