import pathlib
import IPython
import pygmalion as ml
import pandas as pd
import matplotlib.pyplot as plt
from pygmalion.decision_trees import MONOTONICITY
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
ml.datasets.boston_housing(data_path)
df = pd.read_csv(data_path / "boston_housing.csv")
df_train, df_test = ml.split(df, weights=(0.8, 0.2))

# Create and train the model
target = "medv"
inputs = [c for c in df.columns if c != target]
model = ml.decision_trees.GradientBoostingRegressor(inputs, target, {"nox": MONOTONICITY.DECREASING, "tax": MONOTONICITY.INCREASING})
model.fit(df_train, n_trees=100, learning_rate=0.3, max_leaf_count=5)

# Plot validation loss progress
f, ax = plt.subplots()
ax.set_title("Performance on test data against number of trees")
ax.scatter(list(range(1, len(model.trees)+1)), list(ml.RMSE(pred, df_test[target]) for pred in model.predict_partial(df_test)))
ax.set_xlabel("number of trees")
ax.set_ylabel("RMSE")

# Plot results
f, ax = plt.subplots()
ml.plot_fitting(df_train[target], model.predict(df_train), ax=ax, label="training")
ml.plot_fitting(df_test[target], model.predict(df_test), ax=ax, label="testing", color="C3")
ax.set_title(f"R²={ml.R2(model.predict(df_test), df_test[target]):.3g}")
ax.set_xlabel("target")
ax.set_ylabel("predicted")
plt.show()

IPython.embed()