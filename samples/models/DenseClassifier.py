import pathlib
import IPython
import pygmalion as ml
import pygmalion.neural_networks as nn
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
ml.datasets.titanic(data_path)
titanic = pd.read_csv(data_path / "titanic.csv").drop(columns=["PassengerId", "Name", "Ticket"])
df = ml.utilities.embed_categorical(titanic, columns=["Cabin", "Embarked", "Sex"])
df = ml.utilities.mask_nullables(df, ["Age", "Fare"])
df_train, df_val, df_test = ml.utilities.split(df, weights=(0.7, 0.2, 0.1))
target = "Survived"
inputs = [c for c in df.columns if c != target]
classes = df[target].unique()
device = "cuda:0"

# Create and train the model
hidden_layers = [8, 8, 8]
model = nn.DenseClassifier(inputs, target, classes, hidden_layers,
                           activation="elu")
model.to(device)
x_train, y_train = model.data_to_tensor(df_train[inputs], df_train[target])
x_val, y_val = model.data_to_tensor(df_val[inputs], df_val[target])
train_losses, val_losses, grad, best_step = model.fit((x_train, y_train), (x_val, y_val), n_steps=3000, patience=500)

# Plot results
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step)
y_pred, p = model.predict(df_test), model.probabilities(df_test)
f, ax = plt.subplots()
conf = ml.utilities.confusion_matrix(df_test[target], y_pred, classes=classes)
ml.utilities.plot_matrix(conf, ax=ax, cmap="Greens", write_values=True, format=".2%")
acc = ml.utilities.accuracy(y_pred, df_test[target])
ax.set_title(f"Accuracy: {acc:.2%}")
ax.set_ylabel("predicted")
ax.set_xlabel("target")
plt.tight_layout()
plt.show()

IPython.embed()
