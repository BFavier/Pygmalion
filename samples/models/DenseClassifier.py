import pathlib
import IPython
import torch
from typing import Optional
import pygmalion as ml
import pygmalion.neural_networks as nn
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
ml.datasets.iris(data_path)
df = pd.read_csv(data_path / "iris.csv")
df_train, df_val, df_test = ml.split(df, weights=(0.7, 0.2, 0.1))
target = "variety"
inputs = [c for c in df.columns if c != "variety"]
classes = df[target].unique()
device = "cuda:0"

# Defines the batch loader
class Batchifyer:

    def __init__(self, df: pd.DataFrame, batch_size: Optional[int] = None):
        self.x, self.y = model.data_to_tensor(df[inputs], df[target])
        self.batch_size = batch_size

    def __iter__(self):
        if self.batch_size is None:
            # returns the whole dataset as a single batch at each optimization step
            return iter([(self.x, self.y)])
        else:
            # returns a single batch of size batch_size
            indexes = torch.randperm(len(self.x))[:self.batch_size]
            return iter([(self.x[indexes], self.y[indexes])])

# Create and train the model
hidden_layers = [8, 8, 8]
model = nn.DenseClassifier(inputs, target, classes, hidden_layers,
                           activation="elu")
model.to(device)
train_data, val_data = Batchifyer(df_train), Batchifyer(df_val)
train_losses, val_losses, grad, best_step = model.fit(train_data, val_data, n_steps=3000, patience=500)

# Plot results
ml.plot_losses(train_losses, val_losses, grad, best_step)
y_pred, p = model.predict(df_test), model.probabilities(df_test)
f, ax = plt.subplots()
conf = ml.confusion_matrix(df_test[target], y_pred, classes=classes)
ml.plot_matrix(conf, ax=ax, cmap="Greens", write_values=True, format=".2%")
acc = ml.accuracy(y_pred, df_test[target])
ax.set_title(f"Accuracy: {acc:.2%}")
ax.set_ylabel("predicted")
ax.set_xlabel("target")
plt.tight_layout()
plt.show()

IPython.embed()
