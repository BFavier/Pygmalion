import pathlib
import pygmalion.neural_networks as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pygmalion.datasets import titanic
from pygmalion.cross_validation import split
from pygmalion.metrics import confusion_matrix, accuracy
from pygmalion.ploting import plot_matrix, plot_losses
from pygmalion.data_processing import embed_categorical, mask_nullables

plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
titanic(data_path)
data = pd.read_csv(data_path / "titanic.csv").drop(columns=["PassengerId", "Name", "Ticket"])
df = embed_categorical(data, columns=["Cabin", "Embarked", "Sex"], remove_columns=True)
df = mask_nullables(df, ["Age", "Fare"])
df_train, df_val, df_test = split(df, weights=(0.7, 0.2, 0.1))
target = "Survived"
inputs = [c for c in df.columns if c != target]
classes = df[target].unique()
device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu" if torch.cuda.device_count() > 0 else "cpu"

# Create and train the model
model = nn.DenseClassifier(inputs, target, classes, hidden_layers=[8, 8, 8],
                           activation="elu")
model.to(device)
x_train, y_train = model.data_to_tensor(df_train[inputs], df_train[target])
x_val, y_val = model.data_to_tensor(df_val[inputs], df_val[target])
train_losses, val_losses, grad, best_step = model.fit((x_train, y_train), (x_val, y_val), n_steps=3000, patience=500)

# Plot results
plot_losses(train_losses, val_losses, grad, best_step)
y_pred, p = model.predict(df_test), model.probabilities(df_test)
f, ax = plt.subplots()
conf = confusion_matrix(df_test[target], y_pred, classes=classes)
plot_matrix(conf, ax=ax, cmap="Greens", write_values=True, format=".2%")
acc = accuracy(y_pred, df_test[target])
ax.set_title(f"Accuracy: {acc:.2%}")
ax.set_ylabel("predicted")
ax.set_xlabel("target")
plt.tight_layout()
plt.show()
