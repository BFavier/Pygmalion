import pathlib
import IPython
import pygmalion as ml
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / "data"

# Download the data
ml.datasets.boston_housing(data_path)

# Load the data
df = pd.read_csv(data_path / "boston_housing.csv")
target = "medv"
inputs = [c for c in df.columns if c != target]
x = df[inputs]
y = df[target]
data = (x, y)

# Create and train the model
hidden_layers = [{"channels": 8},
                 {"channels": 8}]
model = ml.neural_networks.DenseRegressor(inputs, hidden_layers=hidden_layers,
                                          activation="tanh")
data, test_data = ml.split(data, frac=0.2)
train_data, val_data = ml.split(data, frac=0.2)
model.train(train_data, val_data, patience=100)

# Plot results
model.plot_residuals()
f, ax = plt.subplots()
x, y = train_data
ml.plot_correlation(model(x), y, ax=ax, label="training")
x, y = val_data
ml.plot_correlation(model(x), y, ax=ax, label="validation")
x, y = test_data
ml.plot_correlation(model(x), y, ax=ax, label="testing", color="C3")
ax.set_title(f"R²={ml.R2(model(x), y):.3g}")
plt.show()

IPython.embed()
