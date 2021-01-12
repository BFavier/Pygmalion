import pathlib
import pandas as pd
import machine_learning as ml
import matplotlib.pyplot as plt

plt.style.use("bmh")
package_path = pathlib.Path(__file__).parents[1]
data_path = package_path/"data"

df = pd.read_csv(data_path / "WireBondData.csv", sep=";", decimal=",")
model = ml.LinearModel()
x = ['X1', 'X2', 'X3', 'X4']
y = 'Y'
model.fit_forward(df[x], df[y], p=0.05)
ax = model.plot_fitting(df, df[y])
plt.show()
