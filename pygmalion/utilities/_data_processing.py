import hashlib
from typing import Tuple, Iterable, Optional
import numpy as np
import pandas as pd


def _string_embedding(string: str, n: int=4) -> Tuple[float]:
    """
    Converts a string to a tuple of 'n' floats

    Example:
    --------
    >>> _string_embedding("hello world", 4)
    (0.1640625, 0.6796875, 0.421875, 0.20703125)
    """
    h = list(hashlib.sha1(string.encode("utf-8")).digest()[:n])
    return tuple(i / 256 for i in h)


def embed_categorical(df: pd.DataFrame, dimension: int=4,
                      columns: Optional[Iterable[object]]=None,
                      skip_columns: Optional[Iterable[object]]=[],
                      inplace: bool=False, remove_columns: bool=True) -> pd.DataFrame:
    """
    Converts each categorical columns to 'dimension' additional floating point columns.
    If the list of categorical columns is not supplied, all column that are not floating points are converted.

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to transform
    dimension : int
        the embedding dimension for categorical variables
    columns : iterable of objects, or None
        the list of columns to transform
        defaults to columns that are not floating point if set to None
    skip_columns : iterable of objects, or None
        the column that are in 'skip_columns' are not transformed
    inplace : bool
        If False, the operation is done on a copy of the dataframe
    remove_columns : bool
        If true, the embedded columns are removed
    Example:
    --------
    >>> df = pd.DataFrame([[1.0, 1, "a"], [2.0, 2, "b"]], columns=("floating", "integer", "string"))
    >>> embed_categorical(df, dimension=2)
       floating  integer_1  integer_2  string_1  string_2
    0       1.0   0.207031   0.414062  0.523438  0.964844
    1       2.0   0.851562   0.292969  0.910156  0.839844
    >>> embed_categorical(df, columns=["string"])
       floating  integer  string_1  string_2  string_3  string_4
    0       1.0        1  0.523438  0.964844  0.890625  0.214844
    1       2.0        2  0.910156  0.839844  0.121094  0.367188
    """
    if not inplace:
        df = df.copy()
    if columns is None:
        columns = [c for c, d in df.dtypes.items() if not np.issubdtype(d, np.floating)]
    for col in columns:
        if col in skip_columns:
            continue
        embeddings = tuple(zip(*(_string_embedding(str(v), n=dimension) for v in df[col])))
        for i, e in enumerate(embeddings, start=1):
            df[f"{col}_{i}"] = e
        if remove_columns:
            df.drop(columns=[col], inplace=True)
    return df


def mask_nullables(df: pd.DataFrame, nullable_columns: Optional[Iterable[object]],
                  inplace: bool=False) -> pd.DataFrame:
    """
    Add an '..._is_nan' boolean column for each nullable column and replace nan with 0. in the column.

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to transform
    nullable_columns : iterable of objects, or None
        the list of columns to transform
    inplace : bool
        If False, the operation is done on a copy of the dataframe

    Example:
    --------
    >>> df = pd.DataFrame([[1.0, 1.5], [2.0, float("nan")], [3.0, 3.5]], columns=("A", "B"))
    >>> mask_nullable(df, ["B"])
    >>> mask_nullable(df, ["A", "B"])
    """
    if not inplace:
        df = df.copy()
    for col in nullable_columns:
        df[f"{col}_is_na"] = df[col].isna()
        df[col] = df[col].fillna(0.)
    return df

if __name__ == "__main__":
    import IPython
    IPython.embed()