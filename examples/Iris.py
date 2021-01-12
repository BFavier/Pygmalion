import pathlib
import machine_learning.neural_networks as nn
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")

data_path = pathlib.Path(__file__).parents[1] / "data"

df = pd.read_csv(data_path / "Iris.csv")
x = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["variety"]
model = nn.Classifier()
model.fit(x, y, n_epochs=1000, layers=[16, 16, 16], patience=100)

model.plot_confusion_matrix(x, y)
model.plot_history(log=True)
plt.show()
