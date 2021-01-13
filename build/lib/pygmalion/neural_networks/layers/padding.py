import torch
from typing import Tuple


def load_padding(dump):
    """returns the ConstantPad1d or ConstantPad2d object loaded from a dump"""
    return globals()[dump["type"]].from_dump(dump)


class _ConstantPad:
    """A template for constant pad layers"""

    @classmethod
    def from_dump(cls, dump: dict) -> '_ConstantPad':
        """returns a '_ConstantPad' layer from a dump"""
        assert dump["type"] == cls.__name__
        return cls(dump["padding"], value=dump["value"])

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "padding": list(self.padding),
                "value": self.value}

    def shape_out(self, shape_in: list) -> list:
        c = shape_in[0]
        dims = shape_in[1:]
        before = self.padding[0::2]
        after = self.padding[1::2]
        dims = [b+d+a for d, b, a in
                zip(dims, before, after)]
        return [c] + dims

    def shape_in(self, shape_out: list) -> list:
        c = shape_out[0]
        dims = shape_out[1:]
        before = self.padding[0::2]
        after = self.padding[1::2]
        dims = [-b+d-a for d, b, a in
                zip(dims, before, after)]
        return [c] + dims


class ConstantPad1d(torch.nn.ConstantPad1d, _ConstantPad):
    """A wrapper around torch.nn.ConstantPad1d"""

    def __init__(self, padding: Tuple[int, int], value: float = 0.):
        """
        Parameters
        ----------
        padding : tuple of int
            The (left, right) size of padding
        value : float
            The value of the padding
        """
        torch.nn.ConstantPad1d.__init__(self, padding, value=value)


class ConstantPad2d(torch.nn.ConstantPad2d, _ConstantPad):
    """A wrapper around torch.nn.ConstantPad2d"""

    def __init__(self, padding: Tuple[int, int, int, int], value: float = 0.):
        """
        Parameters
        ----------
        padding : tuple of int
            The (left, right, top, bottom) size of padding
        value : float
            The value of the padding
        """
        torch.nn.ConstantPad2d.__init__(self, padding, value=value)
