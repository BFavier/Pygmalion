import pathlib
import IPython
import pygmalion as ml
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parent / "data"

# Download the data
ml.datasets.boston_housing(data_path)
df = pd.read_csv(data_path / "boston_housing.csv")
df_train, df_val, df_test = ml.split(df, weights=(0.7, 0.2, 0.1))

f, ax = plt.subplots()
ml.plot_matrix(df.corr(), ax=ax, cmap="coolwarm", color_bar=True,
               write_values=True, fontsize=5., vmin=-1., vmax=1., format=".2f")
ax.set_title("Correlation matrix")
plt.show()

# Create and train the model
target = "medv"
inputs = [c for c in df.columns if c != target]
model = ml.neural_networks.DenseRegressor(inputs, target, hidden_layers=[16, 16],
                                          activation="elu")
train_data = model.data_to_tensor(df_train)
val_data = model.data_to_tensor(df_val)
train_losses, val_losses, best_step = model.fit([train_data], [val_data], patience=500)

# Plot losses
f, ax = plt.subplots()
ax.scatter(range(len(train_losses)), train_losses, label="train")
ax.scatter(range(len(val_losses)), val_losses, label="val")
ax.axvline(best_step, 0, 1, color="k", label="best step")
ax.set_yscale("log")
f.legend()
# Plot results
f, ax = plt.subplots()
ml.plot_fitting(df_train[target], model.predict(df_train), ax=ax, label="training")
ml.plot_fitting(df_val[target], model.predict(df_val), ax=ax, label="validation")
ml.plot_fitting(df_test[target], model.predict(df_test), ax=ax, label="testing", color="C3")
ax.set_title(f"R²={ml.R2(model.predict(df_test), df_test[target]):.3g}")
ax.set_xlabel("target")
ax.set_ylabel("predicted")
plt.show()

IPython.embed()
