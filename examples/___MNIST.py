import pathlib
import struct
import machine_learning.neural_networks as nn
import machine_learning as ml
import numpy as np
import matplotlib.pyplot as plt

path = pathlib.Path(__file__).parents[1] / "data" / "MNIST"


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# loading the data
images_data = read_idx(path / "train-images.idx3-ubyte")
images = [ml.Image(data=data) for data in images_data]
labels = read_idx(path / "train-labels.idx1-ubyte")

# training the model
model = nn.ImageClassifier()
windows = [(6, 6), (6, 6)]
channels = [8, 16]
strides = [(3, 3), (1, 1)]
pooling = [(1, 1), (2, 2)]
dense = []
model.fit(images, labels, windows=windows, n_epochs=100, learning_rate=1.0E-1,
          channels=channels, strides=strides, pooling=pooling, dense=dense, l_max=1000)

# ploting confusion matrix
model.plot_history(log=True)
model.plot_confusion_matrix(images, labels)

# ploting some predictions
indexes = np.random.permutation(np.arange(len(labels)))
nx, ny = 5, 5
images_test = [images[indexes[i]] for i in range(nx*ny)]
labels_predicted = model.predict(images_test)
plt.figure()
titles = ["zero", "one", "two", "three", "four", "five",
          "six", "seven", "height", "nine"]
for k in range(0, nx*ny):
    ax = plt.subplot(ny, nx, k+1)
    images_test[k].draw(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles[labels_predicted[k]])
plt.tight_layout()
plt.show()
