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

f, ax = plt.subplots()
ml.plot_matrix(x.corr(), ax=ax, cmap="coolwarm", color_bar=True,
               write_values=True, fontsize=5., vmin=-1., vmax=1., format=".2f")
ax.set_title("Correlation matrix")
plt.show()

# Create and train the model
hidden_layers = [{"features": 16},
                 {"features": 16}]
model = ml.neural_networks.DenseRegressor(inputs, hidden_layers,
                                          activation="elu")
train_data, val_data, test_data = ml.split(x, y, frac=(0.2, 0.1))
model.train(train_data, val_data, patience=500)

# Plot results
model.plot_history()
f, ax = plt.subplots()
x_train, y_train = train_data
ml.plot_fitting(y_train, model(x_train), ax=ax, label="training")
x_val, y_val = val_data
ml.plot_fitting(y_val, model(x_val), ax=ax, label="validation")
x_test, y_test = test_data
ml.plot_fitting(y_test, model(x_test), ax=ax, label="testing", color="C3")
ax.set_title(f"R²={ml.R2(model(x_test), y_test):.3g}")
ax.set_xlabel("target")
ax.set_ylabel("predicted")
plt.show()

IPython.embed()
