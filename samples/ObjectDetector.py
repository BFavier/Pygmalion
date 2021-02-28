import pathlib
import IPython
import json
import numpy as np
import pygmalion as ml
import pygmalion.neural_networks as nn
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / "data" / "aquarium"

# Download the data
ml.datasets.aquarium(data_path.parent)

# Load data
x_train = np.load(data_path / "train_images.npy")
with open(data_path / "train_bounding_boxes.json", "r") as file:
    y_train = json.load(file)
x_val = np.load(data_path / "val_images.npy")
with open(data_path / "val_bounding_boxes.json", "r") as file:
    y_val = json.load(file)
x_test = np.load(data_path / "test_images.npy")
with open(data_path / "test_bounding_boxes.json", "r") as file:
    y_test = json.load(file)
with open(data_path / "class_fractions.json", "r") as file:
    class_fractions = json.load(file)
class_weights = {k: 1/(f+1.0E-8) for k, f in class_fractions.items()}
mean = sum([w for w in class_weights.values()])
class_weights = {k: w/mean for k, w in class_weights.items()}


train_data = (x_train[::100], y_train[::100])
val_data = (x_val[::100], y_val[::100])
classes = [k for k in class_weights.keys()]
boxes_per_cell = 3
in_channels = 3

# Create and train the model
down = [{"window": (3, 3), "channels": 4},
        {"window": (3, 3), "channels": 8},
        {"window": (3, 3), "channels": 16},
        {"window": (3, 3), "channels": 16},
        {"window": (3, 3), "channels": 16},
        {"window": (3, 3), "channels": 16}]
pooling = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
dense = [{"window": (3, 3), "channels": 16},
         {"window": (3, 3), "channels": 32},
         {"window": (3, 3), "channels": 64}]
model = nn.ObjectDetector(in_channels, classes,
                          boxes_per_cell,
                          downsampling=down,
                          pooling=pooling,
                          dense=dense,
                          activation="leaky_relu",
                          GPU=0,
                          learning_rate=1.0E-3,
                          class_weights=class_weights)
model.train(train_data, n_epochs=1000, L_minibatchs=None)

# model.plot_residuals()
f, ax = plt.subplots()
ax.imshow(x_train[0])
# ml.plot_bounding_boxes(y_train[0], ax)
ml.plot_bounding_boxes(model([x_train[0]])[0], ax)
plt.show()

IPython.embed()
