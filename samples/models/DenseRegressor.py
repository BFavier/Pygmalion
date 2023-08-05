import pathlib
import IPython
import pygmalion as ml
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
ml.datasets.boston_housing(data_path)
df = pd.read_csv(data_path / "boston_housing.csv")
df_train, df_val, df_test = ml.utilities.split(df, weights=(0.7, 0.2, 0.1))

# Create and train the model
target = "medv"
inputs = [c for c in df.columns if c != target]
model = ml.neural_networks.DenseRegressor(inputs, target, hidden_layers=[32, 32],
                                          activation="elu", normalize=True, dropout=0.1)
x_train, y_train = model.data_to_tensor(df_train[inputs], df_train[target])
x_val, y_val = model.data_to_tensor(df_val[inputs], df_val[target])
train_losses, val_losses, grad, best_step = model.fit((x_train, y_train), (x_val, y_val), n_steps=5000, patience=500)

# Plot losses
ml.utilities.plot_losses(train_losses, val_losses, grad, best_step)
# Plot results
f, ax = plt.subplots()
ml.utilities.plot_fitting(df_train[target], model.predict(df_train), ax=ax, label="training")
ml.utilities.plot_fitting(df_val[target], model.predict(df_val), ax=ax, label="validation")
ml.utilities.plot_fitting(df_test[target], model.predict(df_test), ax=ax, label="testing", color="C3")
ax.set_title(f"R²={ml.utilities.R2(model.predict(df_test), df_test[target]):.3g}")
ax.set_xlabel("target")
ax.set_ylabel("predicted")
plt.show()

IPython.embed()
