import pathlib
import pygmalion.decision_trees as dt
import pandas as pd
import matplotlib.pyplot as plt

from pygmalion.datasets import iris
from pygmalion.cross_validation import split
from pygmalion.metrics import confusion_matrix, accuracy
from pygmalion.ploting import plot_matrix

plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
iris(data_path)
df = pd.read_csv(data_path / "iris.csv")
df_train, df_test = split(df, weights=(0.8, 0.2))
target = "variety"
inputs = [c for c in df.columns if c != "variety"]
classes = df[target].unique()
device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

# Create and train the model
model = dt.GradientBoostingClassifier(inputs, target, classes)
model.fit(df_train, n_trees=100, learning_rate=0.1, max_leaf_count=4)

# Plot validation loss progress
f, ax = plt.subplots()
ax.set_title("Performance on test data against number of trees")
ax.scatter(list(range(1, len(model.trees)+1)), list(accuracy(pred, df_test[target]) for pred in model.predict_partial(df_test)))
ax.set_xlabel("number of trees")
ax.set_ylabel("accuracy")

# Plot results
y_pred = model.predict(df_test)
f, ax = plt.subplots()
conf = confusion_matrix(df_test[target], y_pred, classes=classes)
plot_matrix(conf, ax=ax, cmap="Greens", write_values=True, format=".2%")
acc = accuracy(y_pred, df_test[target])
ax.set_title(f"Accuracy: {acc:.2%}")
ax.set_ylabel("predicted")
ax.set_xlabel("target")
plt.tight_layout()
plt.show()
