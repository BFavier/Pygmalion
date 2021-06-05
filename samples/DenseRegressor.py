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

f, ax = plt.subplots()
ml.plot_matrix(x.corr(), ax=ax, cmap="coolwarm", color_bar=True,
               write_values=True, fontsize=5., vmin=-1., vmax=1., format=".2f")
ax.set_title("Correlation matrix")
plt.show()

# Create and train the model
hidden_layers = [{"features": 16},
                 {"features": 16},
                 {"features": 16}]
model = ml.neural_networks.DenseRegressor(inputs, hidden_layers,
                                          activation="elu", dropout=0.1)
data, test_data = ml.split(*data, frac=0.2)
train_data, val_data = ml.split(*data, frac=0.2)
model.train(train_data, val_data, patience=500, L2=1.0E-2)

# Plot results
model.plot_history()
f, ax = plt.subplots()
x, y = train_data
ml.plot_fitting(y, model(x), ax=ax, label="training")
x, y = val_data
ml.plot_fitting(y, model(x), ax=ax, label="validation")
x, y = test_data
ml.plot_fitting(y, model(x), ax=ax, label="testing", color="C3")
ax.set_title(f"R²={ml.R2(model(x), y):.3g}")
ax.set_xlabel("target")
ax.set_ylabel("predicted")
plt.show()

IPython.embed()
