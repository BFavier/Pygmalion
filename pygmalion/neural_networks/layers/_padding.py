import torch
from typing import Tuple


class Padding:
    """A template for constant pad layers"""

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        if dump is None:
            return None
        cls = globals()[dump["type"]]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.padding = tuple([size for _, size in dump["margins"].items()])
        obj.value = dump["value"]
        return obj

    @property
    def dump(self) -> dict:
        L = len(self.padding)
        return {"type": type(self).__name__,
                "margins": {side: size for side, size in
                            zip(["left", "right", "top", "bottom"][:L],
                                self.padding)
                            },
                "value": self.value}

    def shape_out(self, shape_in: list) -> list:
        before = self.padding[0::2]
        after = self.padding[1::2]
        return [b+d+a for d, b, a in
                zip(shape_in, before, after)]

    def shape_in(self, shape_out: list) -> list:
        before = self.padding[0::2]
        after = self.padding[1::2]
        return [-b+d-a for d, b, a in
                zip(shape_out, before, after)]


class Padding1d(torch.nn.ConstantPad1d, Padding):

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


class Padding2d(torch.nn.ConstantPad2d, Padding):

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


if __name__ == "__main__":
    import IPython
    IPython.embed()
