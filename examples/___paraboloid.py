import numpy as np
import pandas as pd
import machine_learning.neural_networks as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.style.use("bmh")


def plot_3D(ax, X1, X2, Y):
    ax.plot_surface(X1, X2, Y, cmap=cm.viridis, alpha=0.7, edgecolor="k")
    ax.set_xlabel("X1")
    ax.set_xticks([])
    ax.set_ylabel("X2")
    ax.set_yticks([])
    ax.set_zlabel("Y")
    ax.set_zticks([])


def plot_curve(x, y, func, model):
    fig = plt.figure(figsize=[12, 6])
    X1, X2 = np.meshgrid(np.linspace(-1., 1., 100), np.linspace(-1., 1., 100))
    # plot true function
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    Y_true = np.array([func(x1, x2) for x1, x2 in zip(X1, X2)])
    plot_3D(ax, X1, X2, Y_true)
    ax.set_title("paraboloid")
    # plot model approximation
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    df = pd.DataFrame(np.transpose([X1.flatten(), X2.flatten()]),
                      columns=["X1", "X2"])
    Y_predicted = model.predict(df).reshape(X1.shape)
    plot_3D(ax, X1, X2, Y_predicted)
    # scatter training data
    # ax.scatter(x["X1"], x["X2"], y, c="r", label="Training data")
    # cosmetic
    ax.set_title("neural network model")
    plt.tight_layout()


def paraboloid(X1, X2):
    return X1**2 - X2**2


X1 = np.random.uniform(-1., 1., 200)
X2 = np.random.uniform(-1., 1., 200)
y = [paraboloid(x1, x2) for x1, x2 in zip(X1, X2)]
x = pd.DataFrame(data=np.transpose([X1, X2]), columns=["X1", "X2"])
model = nn.Regressor()
kwargs = {"n_epochs": 500, "layers": (10, 10, 10), "patience": 100,
          "non_linear": "tanh", "L1": 1.0E-2}

model.fit(x, y, **kwargs)
model.plot_history(log=True)
model.plot_fitting(x, y)
plot_curve(x, y, paraboloid, model)

plt.show()
