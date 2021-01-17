import pathlib
import IPython
import pygmalion as ml
import pygmalion.neural_networks as nn
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
model = nn.DistributionNN(inputs, activation="tanh", hidden_layers=[5, 5])
train, val = ml.split(data, frac=0.5)
model.fit(train, val, n_epochs=1000, patience=500)

# Plot results
model.plot_residuals(log=False)
f, ax = plt.subplots()
x, y = train
pred = model(x)
ml.plot_correlation(pred["mean"], y, ax=ax, label="training",
                    s=20/pred["std"])
x, y = val
pred = model(x)
ml.plot_correlation(pred["mean"], y, ax=ax, label="validation",
                    s=20/pred["std"])
plt.show()
IPython.embed()
