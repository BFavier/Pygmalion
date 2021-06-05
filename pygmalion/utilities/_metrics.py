import torch
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Iterable, List, Union


def MSE(predicted: np.ndarray, target: np.ndarray, weights=None):
    """Returns the Mean Squared Error of a regressor"""
    assert len(predicted) == len(target)
    SE = (predicted - target)**2
    if weights is not None:
        SE *= weights
    return np.mean(SE)


def RMSE(predicted: np.ndarray, target: np.ndarray, weights=None):
    """Returns the Root Mean Squared Error of a regressor"""
    return np.sqrt(MSE(predicted, target, weights=weights))


def R2(predicted: np.ndarray, target: np.ndarray, weights=None):
    """Returns the R² score of a regressor"""
    assert len(predicted) == len(target)
    SEres = (predicted - target)**2
    SEtot = (target - np.mean(target))**2
    if weights is not None:
        SEres *= weights
        SEtot *= weights
    return 1 - np.sum(SEres)/np.sum(SEtot)


def accuracy(predicted: np.ndarray, target: np.ndarray):
    """Returns the accuracy of a classifier"""
    assert len(predicted) == len(target)
    return sum([a == b for a, b in zip(predicted, target)])/len(predicted)


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


def confusion_matrix(target: Iterable[str], predicted: Iterable[str],
                     classes: Union[None, List[str]] = None):
    """
    Returns the confusion matrix between prediction and target
    of a classifier

    Parameters
    ----------
    target : iterable of str
        the target to predict
    predicted : iterable of str
        the classes predicted by the model
    classes : None or list of str
        the unique classes to plot
        (can be a subset of the classes in 'predicted' and 'target')
        If None, the classes are infered from unique values from
        'predicted' and 'target'
    """
    assert len(predicted) == len(target)
    if classes is None:
        classes = np.unique(np.stack([predicted, target]))
    predicted = pd.Categorical(predicted, categories=classes)
    target = pd.Categorical(target, categories=classes)
    table = pd.crosstab(predicted, target, normalize="all",
                        rownames=["predicted"], colnames=["target"])
    for c in classes:
        if c not in table.index:
            table.loc[c] = 0
        if c not in table.columns:
            table[c] = 0
    return table.loc[classes[::-1], classes]


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


def GPU_info():
    """
    Returns the list of GPUs, with for each of them:
        * their name
        * their VRAM capacity in GB
        * their current memory usage (in %)
        * their peak memory usage since last call (in %)
    """
    infos = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        max_memory = props.total_memory
        memory_usage = torch.cuda.memory_reserved(i) / max_memory
        max_memory_usage = torch.cuda.max_memory_reserved(i) / max_memory
        infos.append([name, f"{max_memory/1.0E9:.1f} GB",
                      f"{memory_usage*100:.2f}%",
                      f"{max_memory_usage*100:.2f}%"])
        torch.cuda.reset_peak_memory_stats(i)
    df = pd.DataFrame(data=infos, columns=["name", "memory", "usage",
                                           "peak"])
    df.index.name = 'ID'
    return df


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
