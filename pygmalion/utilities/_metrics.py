import torch
import numpy as np
import pandas as pd
from typing import Iterable, List, Optional, Union


def MSE(predicted: np.ndarray, target: np.ndarray, weights=None) -> float:
    """Returns the Mean Squared Error of a regression"""
    assert len(predicted) == len(target)
    SE = (predicted - target)**2
    if weights is not None:
        SE *= weights
    return np.mean(SE)


def RMSE(predicted: np.ndarray, target: np.ndarray, weights=None) -> float:
    """Returns the Root Mean Squared Error of a regression"""
    return np.sqrt(MSE(predicted, target, weights=weights))


def R2(predicted: np.ndarray, target: np.ndarray, weights=None) -> float:
    """Returns the R² score of a regression"""
    assert len(predicted) == len(target)
    SEres = (predicted - target)**2
    SEtot = (target - np.mean(target))**2
    if weights is not None:
        SEres *= weights
        SEtot *= weights
    return 1 - np.sum(SEres)/np.sum(SEtot)


def accuracy(predicted: Iterable, target: Iterable) -> float:
    """
    Returns the accuracy of a classification
    """
    assert len(predicted) == len(target)
    return sum([a == b for a, b in zip(predicted, target)])/len(predicted)


def precision(predicted: Iterable, target: Iterable) -> dict:
    """
    Returns the precision of a classification
    (n correctly predicted as part of the class/n predicted part of the class)
    This is the fraction of true positives amongst predicted positives
    """
    assert len(predicted) == len(target)
    uniques = set(predicted) | set(target)
    precisions = {cat: (sum([p == cat and t == cat
                             for p, t in zip(predicted, target)]) /
                        (sum([p == cat for p in predicted]) + 1.0E-20))
                  for cat in uniques}
    return precisions


def recall(predicted: Iterable, target: Iterable) -> dict:
    """
    Returns the recall of a classification
    (n correctly predicted as part of the class/n part of the class)
    This is the fraction of predicted positives amongst the true positives
    """
    assert len(predicted) == len(target)
    uniques = set(predicted) | set(target)
    precisions = {cat: (sum([p == cat and t == cat
                             for p, t in zip(predicted, target)]) /
                        (sum([t == cat for t in target]) + 1.0E-20))
                  for cat in uniques}
    return precisions


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


def levenshtein(a: Iterable, b: Iterable, max: Optional[int] = None):
    """
    Returns the Levenshtein distance between two iterables (strings or list)
    The Levenshtein distance is the number of insertions/deletions/replacements
    to perform so that the two sequences become identical

    Parameters
    ----------
    a : Iterable
        first sequence
    b : Iterable
        second sequence
    max : int or None
        Once the distance reaches 'max', returns it.
        Usefull to spare some calculation time as complexity is O(n²)

    Returns
    -------
    int :
        the Levenshtein distance
    """
    m = None if max is None else max - 1
    if max == 0:
        return 0
    elif len(b) == 0:
        return len(a)
    elif len(a) == 0:
        return len(b)
    elif a[0] == b[0]:
        return levenshtein(a[1:], b[1:], max=m)
    else:
        return 1 + min(levenshtein(a[1:], b, max=m),
                       levenshtein(a, b[1:], max=m),
                       levenshtein(a[1:], b[1:], max=m))


def WER(predicted: str, target: str, whitespace_separated: bool = True):
    """
    Computes the "Word Error Rate" metric for two texts

    Parameters
    ----------
    predicted : str
        the text predicted by the model
    target : str
        the target text
    whitespace_separated : bool
        If True, the Lenvenshtein distance is applied at word level
        The text is split by white spaces

    Returns
    -------
    float:
        The WER metric
    """
    if whitespace_separated:
        predicted = predicted.split(" ")
        target = target.split(" ")
    return levenshtein(predicted, target)/(len(target) + 1.0E-20)


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
        infos.append([name, f"{max_memory/1024**3:.3g} GB",
                      f"{memory_usage:.2%}",
                      f"{max_memory_usage:.2%}"])
        torch.cuda.reset_peak_memory_stats(i)
    df = pd.DataFrame(data=infos, columns=["name", "memory", "usage",
                                           "peak"])
    df.index.name = 'ID'
    return df


if __name__ == "__main__":
    import IPython
    IPython.embed()
