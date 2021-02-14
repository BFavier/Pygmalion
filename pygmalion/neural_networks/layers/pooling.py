import torch
import torch.nn.functional as F
from typing import Tuple, Union


class Pooling(torch.nn.Module):
    """
    A pooling layer reduces the spatial dimension of a feature map
    by performing max or average pooling over a window for each channel
    """

    @classmethod
    def from_dump(cls, dump: dict) -> object:
        cls = globals()[dump["type"]]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.pooling_window = tuple(dump["pooling window"])
        obj.pooling_type = dump["pooling type"]
        return obj

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "pooling window": [s for s in self.pooling_window],
                "pooling type": self.pooling_type}

    def __init__(self, pooling_window, pooling_type):
        super().__init__()
        self.pooling_type = pooling_type
        self.pooling_window = pooling_window

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.pooling_window is None:
            window = tuple(X.shape[2:])
            return self.function(X, window).view(X.shape[:2])
        else:
            return self.function(X, self.pooling_window)

    def shape_out(self, shape_in: list) -> Union[list, int]:
        if self.pooling_window is None:
            return []
        return [int((d - w)/w + 1) for d, w in
                zip(shape_in, self.pooling_window)]

    def shape_in(self, shape_out: Union[list, int]) -> list:
        if self.pooling_window is None:
            return []
        return [(d-1)*w + w for d, w in
                zip(shape_out, self.pooling_window)]

    @property
    def function(self):
        return self.functions[self.pooling_type]


class Pooling1d(Pooling):

    functions = {"max": F.max_pool1d, "avg": F.avg_pool1d}

    def __init__(self, pooling_window: Union[int, None],
                 pooling_type: str = "max"):
        """
        Parameters:
        -----------
        pooling_window : int or None
            The size of the pooling window
            Or None if overall pool
        pooling_type : one of {"max", "avg"}
            The type of pooling to perform
        """
        super().__init__(pooling_window, pooling_type)


class Pooling2d(Pooling):

    functions = {"max": F.max_pool2d, "avg": F.avg_pool2d}

    def __init__(self, pooling_window: Union[Tuple[int, int], None],
                 pooling_type: str = "max"):
        """
        Parameters:
        -----------
        pooling_window : tuple of int or None
            The (height, width) of the pooling window
            Or None if overall pool
        pooling_type : one of {"max", "avg"}
            The type of pooling to perform
        """
        super().__init__(pooling_window, pooling_type)
