import torch
import torch.nn.functional as F
from typing import Tuple, List, Union


# def load_pooling(dump):
#     """returns the Pooling object loaded from a dump"""
#     return globals()[dump["type"]].from_dump(dump)


# class _Pooling:
#     """A template for pooling layers"""

#     @classmethod
#     def from_dump(cls, dump: dict) -> '_Pooling':
#         """returns a '_Pooling' layer from a dump"""
#         assert dump["type"] == cls.__name__
#         return cls(dump["kernel size"], stride=dump["stride"])

#     @property
#     def dump(self) -> dict:
#         ks = self.kernel_size
#         st = self.stride
#         return {"type": type(self).__name__,
#                 "kernel size": list(ks) if hasattr(ks, "__iter__") else ks,
#                 "stride": list(st) if hasattr(st, "__iter__") else st}

#     def shape_out(self, shape_in: list) -> list:
#         c = shape_in[0]
#         dims = shape_in[1:]
#         dims = [int((d - k)/s + 1) for d, k, s in
#                 zip(dims, self.kernel_size, self.stride)]
#         return [c] + dims

#     def shape_in(self, shape_out: list) -> list:
#         c = shape_out[0]
#         dims = shape_out[1:]
#         dims = [(d-1)*s + k for d, k, s in
#                 zip(dims, self.kernel_size, self.stride)]
#         return [c] + dims


# class MaxPool1d(torch.nn.MaxPool1d, _Pooling):
#     """A wrapper around torch.nn.MaxPool1d"""

#     def __init__(self, kernel_size: int, stride: int = 1):
#         """
#         Parameters
#         ----------
#         kernel_size : int
#             The size of the pooling window
#         stride : int
#             The stride of the pooling window
#         """
#         torch.nn.MaxPool1d.__init__(self, kernel_size, stride)


# class MaxPool2d(torch.nn.MaxPool2d, _Pooling):
#     """A wrapper around torch.nn.MaxPool2d"""

#     def __init__(self, kernel_size: Tuple[int, int], stride: int = 1):
#         """
#         Parameters
#         ----------
#         kernel_size : tuple of int
#             The (height, width) size of the pooling window
#         stride : tuple of int
#             The (height, width) stride of the pooling window
#         """
#         torch.nn.MaxPool1d.__init__(self, kernel_size, stride)


# class AvgPool1d(torch.nn.AvgPool1d, _Pooling):
#     """A wrapper around torch.nn.AvgPool1d"""

#     def __init__(self, kernel_size: int, stride: int = 1):
#         """
#         Parameters
#         ----------
#         kernel_size : int
#             The size of the pooling window
#         stride : int
#             The stride of the pooling window
#         """
#         torch.nn.AvgPool1d.__init__(self, kernel_size, stride)


# class AvgPool2d(torch.nn.AvgPool2d, _Pooling):
#     """A wrapper around torch.nn.AvgPool2d"""

#     def __init__(self, kernel_size: Tuple[int, int], stride: int = 1):
#         """
#         Parameters
#         ----------
#         kernel_size : tuple of int
#             The (height, width) size of the pooling window
#         stride : tuple of int
#             The (height, width) stride of the pooling window
#         """
#         torch.nn.AvgPool2d.__init__(self, kernel_size, stride)


# class _OverallPool(torch.nn.Module):
#     """
#     Over all pooling transforms a tensor of shape (N, C, ...)
#     to a tensor of shape (N, C)
#     """

#     @classmethod
#     def from_dump(cls, dump: dict) -> 'OverallPool1d':
#         assert dump["type"] == cls.__name__
#         return cls(dump["pooling type"])

#     def __init__(self, pooling_type: str, dimension: str):
#         """
#         Parameters
#         ----------
#         pooling type : one of {"Max", "Avg"}
#             the type of pooling to perform
#         dimension : one of {"1d", "2d"}
#             the dimension of the tensor to treat
#         """
#         super().__init__()
#         self.pooling_type = pooling_type
#         self.func = getattr(F, pooling_type.lower()+"_pool"+dimension)

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         return self.func(X, X.shape[2:]).view(X.shape[:2])

#     @property
#     def dump(self):
#         return {"type": type(self).__name__,
#                 "pooling type": self.pooling_type}

#     def shape_in(self, shape_out: int) -> List[int]:
#         return [shape_out, 1, 1]

#     def shape_out(self, shape_in: List[int]) -> int:
#         return shape_in[0]


# class OverallPool1d(_OverallPool):

#     def __init__(self, pooling_type: str):
#         """
#         Parameters
#         ----------
#         pooling type : one of {"Max", "Avg"}
#             the type of pooling to perform
#         """
#         super().__init__(pooling_type, "1d")


# class OverallPool2d(_OverallPool):

#     def __init__(self, pooling_type: str):
#         """
#         Parameters
#         ----------
#         pooling type : one of {"Max", "Avg"}
#             the type of pooling to perform
#         """
#         super().__init__(pooling_type, "2d")


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
