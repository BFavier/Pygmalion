import pathlib
import IPython
import numpy as np
import pygmalion as ml
import pygmalion.neural_networks as nn
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / ".." / "data" / "Fashion-MNIST"


# Load data
with open(data_path / "categories.txt", "r") as file:
    data = file.read()
categories = data.split(",")
x_train = np.load(data_path / "train_images.npy")
y_train = np.array(categories)[np.load(data_path / "train_labels.npy")]
x_test = np.load(data_path / "test_images.npy")
y_test = np.array(categories)[np.load(data_path / "test_labels.npy")]
in_channels = 1

# Create and train the model
convolutions = [{"window": (4, 4), "channels": 5},
                {"window": (4, 4), "channels": 10},
                {"window": (3, 3), "channels": 20}]
pooling = [(2, 2), (2, 2), (2, 2)]
model = nn.ImageClassifier(in_channels, categories,
                           convolutions=convolutions,
                           pooling=pooling,
                           fully_connected=[],
                           activation="leaky_relu",
                           padded=False,
                           GPU=True,
                           learning_rate=1.0E-2)
# print(model.module.shapes)
train, val = ml.split((x_train, y_train), frac=0.2)
model.fit(train, val, n_epochs=500, L_minibatchs=3000)

# Plot results
model.plot_residuals()
f, ax = plt.subplots()
y_pred = model(x_test)
ml.plot_confusion_matrix(y_pred, y_test, ax=ax)
acc = ml.accuracy(y_pred, y_test)
ax.set_title(f"Accuracy = {acc:.3g}")
f.tight_layout()

# Plot metrics
lx, ly = 5, 5
x = x_test[np.random.permutation(len(x_test))[:lx*ly]]
y = model(x)
f, axes = plt.subplots(figsize=[8, 8], ncols=lx, nrows=ly)
for n in range(lx*ly):
    i = n // lx
    j = n % lx
    ax = axes[i, j]
    ax.imshow(x[n], cmap="Greys")
    ax.set_title(y[n])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
f.tight_layout()

plt.show()
IPython.embed()
