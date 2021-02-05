import pathlib
import IPython
import pygmalion as ml
import pygmalion.neural_networks as nn
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / ".." / "data"

# Load the data
df = pd.read_csv(data_path / "iris.csv")
target = "variety"
inputs = [c for c in df.columns if c != "variety"]
x = df[inputs]
y = df[target]
classes = y.unique()

data, test_data = ml.split((x, y), frac=0.2)

# Create and train the model
hidden_layers = [{"channels": 5},
                 {"channels": 5},
                 {"channels": 5}]
model = nn.DenseClassifier(inputs, classes, hidden_layers=hidden_layers,
                           activation="elu")
train_data, val_data = ml.split(data, frac=0.1)
model.train(train_data, val_data, n_epochs=2000)

# Plot results
model.plot_residuals()
x, y = test_data
y_pred = model(x)
f, ax = plt.subplots()
ml.plot_confusion_matrix(y_pred, y, ax=ax)
acc = ml.accuracy(y_pred, y)*100
ax.set_title(f"Accuracy: {acc:.2f}%")
plt.tight_layout()
plt.show()

IPython.embed()
