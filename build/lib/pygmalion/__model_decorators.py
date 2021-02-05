import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from .utilities import MSE, RMSE, R2, accuracy, plot_correlation


def regressor(cls):

    class Regressor(cls):

        def __init__(self, *args, **kwargs):
            """
            """
            super().__init__(*args, **kwargs)

        def plot_correlation(self, x, y, weights=None, ax=None, label="_"):
            """
            """
            y_pred = self(x)
            ax.scatter(y, y_pred, marker=".", s=weights, label=label)
            points = np.concatenate([c.get_offsets() for c in ax.collections])
            inf, sup = points.min(), points.max()
            delta = sup - inf if sup != inf else 1
            sup += 0.01*delta
            inf -= 0.01*delta
            plt.plot([inf, sup], [inf, sup], color="k", zorder=0)
            ax.set_xlim([inf, sup])
            ax.set_ylim([inf, sup])
            ax.set_xlabel("target")
            ax.set_ylabel("predicted")
            ax.set_aspect("equal", "box")
            ax.legend()

    return Regressor


def classifier(cls):

    class Classifier(cls):

        def __init__(self, *args, **kwargs):
            """
            """
            super().__init__(*args, **kwargs)

        def plot_confusion_matrix(self, x, y, ax=None, cmap="Blues"):
            """
            """
            indexes = range(len(self.classes))
            if ax is None:
                f, ax = plt.subplots()
            y_pred = pd.Series(self.index(x))
            y_target = pd.Series([self.classes.index(c) for c in y])
            tab = pd.crosstab(y_pred, y_target, normalize="all")
            for i in indexes:
                if i not in tab.index:
                    tab.loc[i] = 0
                if i not in tab.columns:
                    tab[i] = 0
            tab = tab.to_numpy()
            ax.imshow(tab, origin="lower", interpolation="nearest", cmap=cmap)
            ax.grid(False)
            ax.set_xticks(indexes)
            ax.set_xticklabels(self.classes)
            ax.set_xlabel("target")
            ax.set_yticks(indexes)
            ax.set_yticklabels(self.classes)
            ax.set_ylabel("predicted")
            for y in indexes:
                for x in indexes:
                    val = tab[y, x]
                    if val >= 0.01:
                        ax.text(x, y, f"{val:.2f}", va='center', ha='center')

    return Classifier
