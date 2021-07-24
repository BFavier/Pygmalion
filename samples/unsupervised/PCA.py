import pygmalion as ml
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import IPython

data_path = pathlib.Path(__file__).parents[1] / "data"
ml.datasets.boston_housing(data_path)
df = pd.read_csv(data_path / "boston_housing.csv")
x = df[[c for c in df.columns if c != "medv"]]
pca = ml.unsupervised.PCA()
pca.train(x)
pca.plot_explained_variance()
plt.show()
res = pca(x)


IPython.embed()
