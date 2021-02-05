import pathlib
import IPython
import numpy as np
import pygmalion as ml
import pygmalion.neural_networks as nn
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / ".." / "data" / "Fashion-MNIST"


# Load data
with open(data_path / "classes.txt", "r") as file:
    data = file.read()
classes = data.split(",")
x_train = np.load(data_path / "train_images.npy")
y_train = np.array(classes)[np.load(data_path / "train_labels.npy")]
x_test = np.load(data_path / "test_images.npy")
y_test = np.array(classes)[np.load(data_path / "test_labels.npy")]
in_channels = 1

# Create and train the model
convolutions = [[{"window": (3, 3), "channels": 8, "padded": False},
                 {"window": (3, 3), "channels": 8}],
                [{"window": (3, 3), "channels": 16, "padded": False, "dropout": 0.2},
                 {"window": (3, 3), "channels": 16, "dropout": 0.2}],
                [{"window": (3, 3), "channels": 32, "padded": False, "dropout": 0.2},
                 {"window": (3, 3), "channels": 32, "dropout": 0.2}]]
pooling = [(2, 2), (2, 2), (2, 2)]
dense = [{"channels": 16, "dropout": 0.2}]
model = nn.ImageClassifier(in_channels, classes,
                           convolutions=convolutions,
                           pooling=pooling,
                           dense=dense,
                           activation="leaky_relu",
                           GPU=True,
                           learning_rate=1.0E-2)
# print(model.module.shapes)
train_data, val_data = ml.split((x_train, y_train), frac=0.2)
model.train(train_data, val_data, n_epochs=300, L_minibatchs=1000)
model.learning_rate = 5.0E-3
model.train(train_data, val_data, n_epochs=300, L_minibatchs=1000)

# Plot results
model.plot_residuals()
f, ax = plt.subplots()
y_pred = sum([model(x_test[i:i+100]) for i in range(0, len(x_test), 100)], [])
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
