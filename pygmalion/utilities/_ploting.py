import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Iterable, Union


def plot_correlation(target: Iterable[float], predicted: Iterable[float],
                     ax: Union[None, matplotlib.axes.Axes] = None,
                     label: str = "_",
                     **kwargs):
    """
    Plots the correlation between prediction and target of a regressor

    Parameters
    ----------
    target : iterable of str
        the target to predict
    predicted : iterable of str
        the classes predicted by the model
    ax : None or matplotlib.axes.Axes
        The axes to draw on. If None a new window is created.
    label : str
        The legend of the data plotted. Ignored if starts with '_'.
    **kwargs : dict
        dict of keywords passed to 'plt.scatter'
    """
    assert len(predicted) == len(target)
    if ax is None:
        f, ax = plt.subplots(figsize=[5, 5])
    ax.scatter(target, predicted, label=label, **kwargs)
    points = np.concatenate([c.get_offsets() for c in ax.collections])
    inf, sup = points.min(), points.max()
    delta = sup - inf if sup != inf else 1
    sup += 0.05*delta
    inf -= 0.05*delta
    ax.plot([inf, sup], [inf, sup], color="k", zorder=0)
    ax.set_xlim([inf, sup])
    ax.set_ylim([inf, sup])
    ax.set_aspect("equal", "box")
    legend = ax.legend()
    if len(legend.texts) == 0:
        legend.remove()


def plot_bounding_boxes(bboxes: dict, ax: matplotlib.axes.Axes,
                        class_colors: dict = {}, color: str = "r",
                        label_class: bool = True):
    """
    plot the image with given bounding boxes

    Parameters
    ----------
    bounding_boxes : dict
        A dict containing the following keys:
        * x1, y1, x2, y2 : list of int
            The coordinates in pixel of each bboxe corner
        * class : list of str
            The name of the class predicted for eahc bboxe
        * [confidence : list of float]
            The optional confidence of the bounding boxe
    ax : matplotlib.axes.Axes
        The matplotlib axes to draw on
    class_colors : dict
        A dictionary of {class: color} for the color of the boxes
        Can be any color format supported by matplotlib
    color : str or list
        the default color for classes that are not present in class_colors
    """
    coords = zip(bboxes["x1"], bboxes["y1"], bboxes["x2"], bboxes["y2"])
    for i, (x1, y1, x2, y2) in enumerate(coords):
        boxe_class = bboxes["class"][i]
        boxe_color = class_colors.get(boxe_class, color)
        xinf, yinf = min(x1, x2), min(y1, y2)
        w, h = abs(x2-x1), abs(y2-y1)
        rect = patches.Rectangle((xinf, yinf), w, h,
                                 linewidth=1, edgecolor=boxe_color,
                                 facecolor='none')
        if label_class:
            s = boxe_class
            confidence = bboxes.get("confidence", None)
            if confidence is not None:
                s += f": {confidence[i]*100:.1f}%"
            ax.text(xinf, yinf-1, s, color=boxe_color)
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_matrix(table: pd.DataFrame,
                ax: Union[None, matplotlib.axes.Axes] = None,
                cmap: str = "Greens"):
    """
    Plots the confusion matrix between prediction and target
    of a classifier

    Parameters
    ----------
    table : pd.DataFrame
        The matrix to plot the content of
    ax : None or matplotlib.axes.Axes
        The axis to draw on. If None, a new window is created.
    cmap : str
        The name of the maplotlib colormap
    """
    if ax is None:
        f, ax = plt.subplots()
    inf, sup = table.min().min(), table.max().max()
    ax.imshow(table.to_numpy(), interpolation="nearest",
              cmap=cmap, vmin=inf, vmax=sup)
    ax.grid(False)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, rotation=45)
    for y, cy in enumerate(table.index):
        for x, cx in enumerate(table.columns):
            val = table.loc[cy, cx]
            if val >= 0.01:
                color = "white" if (val - inf)/(sup - inf) > 0.5 else "black"
                ax.text(x, y, f"{val:.2f}", va='center', ha='center',
                        color=color)
