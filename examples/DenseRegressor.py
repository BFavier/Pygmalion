import pathlib
import IPython
import pygmalion as ml
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / ".." / "data"

# Load the data
df = pd.read_csv(data_path / "boston_housing.csv")
target = "medv"
inputs = [c for c in df.columns if c != target]
x = df[inputs]
y = df[target]
data = (x, y)

# Create and train the model
model = ml.neural_networks.DenseRegressor(inputs, activation="tanh")
train, val = ml.split(data, frac=0.2)
model.fit(train, val)

# Plot results
model.plot_residuals()
f, ax = plt.subplots()
x, y = train
ml.plot_correlation(model(x), y, ax=ax, label="training")
x, y = val
ml.plot_correlation(model(x), y, ax=ax, label="validation")
plt.show()

IPython.embed()