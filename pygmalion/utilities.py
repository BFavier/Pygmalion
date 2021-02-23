import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Tuple, Iterable, List, Union


def split(data: Tuple[Any], frac: float = 0.2, shuffle: bool = True) -> tuple:
    """
    Splits the input data in two (train, test)

    Parameters
    ----------
    data : tuple
        Tuple of iterables
    frac : float
        The fraction of testing data
    shuffle : bool
        If True, the data is shuffled before splitting

    Returns
    -------
    tuple :
        the 'first' and 'second' tuples of data
    """
    L = len(data[0])
    indexes = np.random.permutation(L) if shuffle else np.arange(L)
    limit = int(round(frac * L))
    b = indexes[:limit]
    a = indexes[limit:]
    train = [_index(d, a) for d in data]
    test = [_index(d, b) for d in data]
    return tuple(train), tuple(test)


def kfold(data: Tuple[Any], k: int = 3, shuffle: bool = True) -> tuple:
    """
    Splits the input data into k-folds of (train, test) data

    Parameters
    ----------
    data : tuple
        Tuple of iterables
    k : int
        The number of folds to yield
    shuffle : bool
        If True, the data is shuffled before splitting


    Yields
    ------
    tuple :
        the (train, test) tuple of data
    """
    L = len(data[0])
    indexes = np.random.permutation(L) if shuffle else np.arange(L)
    indexes = np.split(indexes, k)
    for i in range(k):
        train = []
        for j, ind in enumerate(indexes):
            if j == i:
                test = ind
            else:
                train.extend(ind)
        yield _index(data, train), _index(data, test)


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


def plot_correlation(predicted: Iterable[float], target: Iterable[float],
                     ax: Union[None, matplotlib.axes.Axes] = None,
                     label: str = "_",
                     **kwargs):
    """
    Plots the correlation between prediction and target of a regressor

    Parameters
    ----------
    predicted : iterable of str
        the classes predicted by the model
    target : iterable of str
        the target to predict
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
    plt.plot([inf, sup], [inf, sup], color="k", zorder=0)
    ax.set_xlim([inf, sup])
    ax.set_ylim([inf, sup])
    ax.set_xlabel("target")
    ax.set_ylabel("predicted")
    ax.set_aspect("equal", "box")
    ax.legend()


def confusion_matrix(predicted: Iterable[str], target: Iterable[str],
                     classes: Union[None, List[str]] = None):
    """
    Returns the confusion matrix between prediction and target
    of a classifier

    Parameters
    ----------
    predicted : iterable of str
        the classes predicted by the model
    target : iterable of str
        the target to predict
    classes : None or list of str
        the unique classes to plot
        (can be a subset of the classes in 'predicted' and 'target')
        If None, the classes are infered from unique values from
        'predicted' and 'target'
    """
    assert len(predicted) == len(target)
    if classes is None:
        classes = np.unique(np.stack([predicted, target]))
    predicted = pd.Series(predicted).reset_index(drop=True)
    target = pd.Series(target).reset_index(drop=True)
    tab = pd.crosstab(predicted, target, normalize="all")
    for c in classes:
        if c not in tab.index:
            tab.loc[c] = 0
        if c not in tab.columns:
            tab[c] = 0
    return tab.loc[classes, classes]


def plot_confusion_matrix(*args, ax: Union[None, matplotlib.axes.Axes] = None,
                          cmap: str = "Greens", **kwargs):
    """
    Plots the confusion matrix between prediction and target
    of a classifier

    Parameters
    ----------
    *args : tuple
        args passed to 'confusion_matrix'
    ax : None or matplotlib.axes.Axes
        The axis to draw on. If None, a new window is created.
    cmap : str
        The name of the maplotlib colormap
    **kwargs : dict
        kwargs passed to 'confusion_matrix'
    """
    if ax is None:
        f, ax = plt.subplots()
    tab = confusion_matrix(*args, **kwargs)
    inf, sup = tab.min().min(), tab.max().max()
    ax.imshow(tab.to_numpy(), origin="lower", interpolation="nearest",
              cmap=cmap, vmin=inf, vmax=sup)
    ax.grid(False)
    ax.set_xticks(range(len(tab.columns)))
    ax.set_xticklabels(tab.columns, rotation=45)
    ax.set_xlabel("target")
    ax.set_yticks(range(len(tab.index)))
    ax.set_yticklabels(tab.index, rotation=45)
    ax.set_ylabel("predicted")
    for y, cy in enumerate(tab.index):
        for x, cx in enumerate(tab.columns):
            val = tab.loc[cy, cx]
            if val >= 0.01:
                color = "white" if (val - inf)/(sup - inf) > 0.5 else "black"
                ax.text(x, y, f"{val:.2f}", va='center', ha='center',
                        color=color)


def GPU_info():
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
    df = pd.DataFrame(data=infos, columns=["name", "memory", "memory use",
                                           "peak memory use"])
    df.index.name = 'ID'
    return df


def _index(data: Any, at: np.ndarray):
    """Indexes an input data. Method depends on it's type"""
    if data is None:
        return None
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.iloc[at]
    elif isinstance(data, np.ndarray):
        return data[at]
    elif isinstance(data, list):
        return [data[i] for i in at]
    else:
        raise RuntimeError(f"'{type(data)}' can't be indexed")


if __name__ == "__main__":
    import IPython
    IPython.embed()
