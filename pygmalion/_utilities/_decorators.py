"""
This module contains decorators
"""
import inspect
from types import FunctionType
from itertools import chain
from typing import Tuple


def document(*wrapped: Tuple[FunctionType]) -> FunctionType:
    """
    Changes inplace the signature and doc of the wrapper
    to integrate those of the wrapped functions.

    Example
    -------
    >>> def f(x, y=3):
    ...     "f is a wrapped function"
    ...     pass
    ...
    >>> def g(*args, **kwargs):
    ...     "g is a wrapper function"
    ...     pass
    ...
    >>> @document(f)
    >>> def h(*args, **kwargs):
    ...     "\nh is a wrapper function"
    ...     pass
    ...
    >>> help(f)
    f(x, y=3)
        f is a wrapped function
    >>> print(help(g))
    g(*args, **kwargs)
        g is a wrapper function
    >>> help(h)
    h(x, y=3)
        f is a wrapped function
        h is a wrapper function
    """
    def decorator(wrapper: FunctionType) -> FunctionType:
        # a target is used is used to handle 'properties' and 'bound methods'
        target = wrapper.__func__ if hasattr(wrapper, "__func__") else wrapper
        target.__signature__ = _mixed_signature(wrapper, *wrapped)
        target.__doc__ = _mixed_doc(wrapper, *wrapped)
        return wrapper
    return decorator


def _mixed_signature(wrapper: FunctionType, *wrapped: Tuple[FunctionType]
                     ) -> inspect.Signature:
    """
    This is a tool to overwrite the __signature__ of a wrapper function
    so that the args and kwargs of the wrapped function appear in the signature
    in place of *args and **kwargs

    Example
    -------
    >>> def f(i: int, *args, j: float = 3.0, **kwargs):
    ...     return i*j
    ...
    >>> def g(x, *args, y=4.0, **kwargs):
    ...     return f(*args, **kwargs)**x + y
    ...
    >>> _mixed_signature(f, g)
    <Signature (i: int, x, *args, j: float = 3.0, y=4.0, **kwargs)>
    """
    p1 = sum((list(inspect.signature(w).parameters.values()) for w in wrapped),
             [])
    p2 = list(inspect.signature(wrapper).parameters.values())
    positional = (inspect.Parameter(p.name,
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                    annotation=p.annotation)
                  for p in chain(p1, p2)
                  if p.kind == inspect.Parameter.POSITIONAL_ONLY
                  or (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                      and p.default == inspect.Parameter.empty))
    keywords = (inspect.Parameter(p.name,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  default=p.default, annotation=p.annotation)
                for p in chain(p1, p2)
                if p.kind == inspect.Parameter.KEYWORD_ONLY
                or (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                    and p.default != inspect.Parameter.empty))
    parameters = chain(positional, keywords)
    parameters = {p.name: p for p in parameters}.values()
    return inspect.Signature(parameters=parameters)


def _mixed_doc(wrapper: FunctionType, *wrapped: Tuple[FunctionType]) -> str:
    """
    Returns the doc of the wrapper appended at the end of the doc of the
    wrapped function
    """
    d1 = "".join(w.__doc__ or "" for w in wrapped)
    d2 = wrapper.__doc__ or ""
    return d1+d2


if __name__ == "__main__":
    import IPython

    def f(x, y=3):
        "f is a wrapped function"
        pass

    def g(*args, **kwargs):
        "\ng is a wrapper function"
        pass

    @document(f, g)
    def h(*args, **kwargs):
        "\nh is a wrapper function"
        pass

    IPython.embed()
