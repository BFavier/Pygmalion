import pathlib
import IPython
import pygmalion as ml
import pygmalion.decision_trees as dt
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
ml.datasets.iris(data_path)
df = pd.read_csv(data_path / "iris.csv")
df_train, df_test = ml.utilities.split(df, weights=(0.8, 0.2))
target = "variety"
inputs = [c for c in df.columns if c != "variety"]
classes = df[target].unique()
device = "cuda:0"

# Create and train the model
model = dt.DecisionTreeClassifier(inputs, target, classes)
model.fit(df_train, max_leaf_count=10)

# Plot results
y_pred = model.predict(df_test)
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
