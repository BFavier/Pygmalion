import pathlib
import pygmalion.decision_trees as dt
import pandas as pd
import matplotlib.pyplot as plt
from pygmalion.decision_trees import MONOTONICITY
from pygmalion.datasets import boston_housing
from pygmalion.cross_validation import split
from pygmalion.metrics import R2
from pygmalion.ploting import plot_fitting
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
boston_housing(data_path)
df = pd.read_csv(data_path / "boston_housing.csv")
df_train, df_test = split(df, weights=(0.8, 0.2))

# Create and train the model
target = "medv"
inputs = [c for c in df.columns if c != target]
model = dt.DecisionTreeRegressor(inputs, target, {"nox": MONOTONICITY.DECREASING, "tax": MONOTONICITY.INCREASING})
model.fit(df_train, max_leaf_count=10)

# Plot results
f, ax = plt.subplots()
plot_fitting(df_train[target], model.predict(df_train), ax=ax, label="training")
plot_fitting(df_test[target], model.predict(df_test), ax=ax, label="testing", color="C3")
ax.set_title(f"R²={R2(model.predict(df_test), df_test[target]):.3g}")
ax.set_xlabel("target")
ax.set_ylabel("predicted")
plt.show()
