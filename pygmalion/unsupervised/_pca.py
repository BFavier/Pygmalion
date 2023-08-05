import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from pygmalion._model import Model


class PCA(Model):

    @classmethod
    def from_dump(cls, dump: dict) -> "PCA":
        """
        """
        assert dump["type"] == cls.__name__
        projection = {k: (np.array(v[0], dtype=float), v[1]) for k, v in
                      dump["projection"].items()}
        return PCA(dump["offset"], dump["scale"], projection)

    def __init__(self,
                 offset: Dict[str, float] = {}, scale: Dict[str, float] = {},
                 projection: Dict[str, Tuple[np.ndarray, float]] = {}):
        self.offset = pd.Series(offset, dtype=float)
        self.scale = pd.Series(scale, dtype=float)
        self.projection = projection

    def __repr__(self):
        return "PCA("+", ".join(self.projection.keys())+")"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe of the projection onto the PCA coordinates
        of each observation
        """
        x = df[self.offset.index] - self.offset
        if self.scale is not None:
            x = x/self.scale
        columns = self.projection.keys()
        matrix = np.stack([self.projection[c][0] for c in columns]).T
        pca = x.to_numpy() @ matrix
        return pd.DataFrame(data=pca, columns=columns)

    def fit(self, df: pd.DataFrame, scale: bool = True):
        """
        Compute the principal components of a points cloud

        Parameters
        ----------
        df : pd.DataFrame
            dataframe of numerical values, each column is a variable
        normalize : bool
            If True the data is normalized as (x - mean(x))/std(x)
            If False the data is only translated as (x - mean(x))
        """
        self.offset = df.mean()
        self.scale = df.std() if scale else None
        df = df - self.offset
        if scale:
            df = df/self.scale
        covariance_matrix = np.cov(df.to_numpy().T)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        projections = sorted([(vector, value) for vector, value in
                              zip(eigen_vectors, eigen_values)],
                             key=lambda x: x[1], reverse=True)
        self.projection = {f"PCA{i+1}": (vector, value) for i, (vector, value)
                           in enumerate(projections)}

    def plot_explained_variance(self, ax: Optional[plt.Axes] = None):
        """
        Plot the explained variance as a function of the number of components
        that are kept
        """
        if ax is None:
            f, ax = plt.subplots()
        x = list(range(len(self.projection)+1))
        eigen = [v[1] for v in self.projection.values()]
        total = sum(eigen)
        y = [sum(eigen[:i])/total for i in x]
        ax.plot(x, y)
        ax.set_ylabel("explained variance")
        ax.set_xlabel("number of components")

    @property
    def dump(self) -> dict:
        offset = dict(self.offset)
        scale = dict(self.scale) if self.scale is not None else self.scale
        projection = {k: [v[0].tolist(), v[1]] for k, v in
                      self.projection.items()}
        return {"type": type(self).__name__,
                "offset": offset,
                "scale": scale,
                "projection": projection}
