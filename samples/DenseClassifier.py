import pathlib
import IPython
import pygmalion as ml
import pygmalion.neural_networks as nn
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / "data"

# Download the data
ml.datasets.iris(data_path)

# Load the data
df = pd.read_csv(data_path / "iris.csv")
target = "variety"
inputs = [c for c in df.columns if c != "variety"]
x = df[inputs]
y = df[target]
classes = y.unique()

# Create and train the model
hidden_layers = [{"features": 8},
                 {"features": 8},
                 {"features": 8}]
model = nn.DenseClassifier(inputs, classes, hidden_layers,
                           activation="elu")
train_data, val_data, test_data = ml.split(x, y, frac=(0.1, 0.2))
model.train(train_data, val_data, n_epochs=3000, patience=200, L2=0.001)

# Plot results
model.plot_history()
x_test, y_test = test_data
y_pred = model(x_test)
f, ax = plt.subplots()
ml.plot_matrix(ml.confusion_matrix(y_test, y_pred, classes=classes), ax=ax,
               cmap="Greens", write_values=True, format=".2%")
acc = ml.accuracy(y_pred, y_test)
ax.set_title(f"Accuracy: {acc:.2%}")
ax.set_ylabel("predicted")
ax.set_xlabel("target")
plt.tight_layout()
plt.show()

IPython.embed()
