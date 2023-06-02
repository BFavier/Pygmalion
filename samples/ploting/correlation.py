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
df_train, df_val, df_test = ml.split(df, weights=(0.7, 0.2, 0.1))

# Plot the correlation matrix between data
f, ax = plt.subplots()
ml.plot_matrix(df.corr(), ax=ax, cmap="coolwarm", color_bar=True,
               write_values=True, fontsize=5., vmin=-1., vmax=1., format=".2f")
ax.set_title("Correlation matrix")
plt.show()