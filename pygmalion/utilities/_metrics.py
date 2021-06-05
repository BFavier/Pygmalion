import torch
import numpy as np
import pandas as pd
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
