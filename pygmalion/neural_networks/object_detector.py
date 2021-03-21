import torch
import numpy as np
from typing import Union, List, Tuple
from .layers import Conv2d, BatchNorm2d
from .layers import Encoder2d, Dense2d
from .conversions import bounding_boxes_to_tensor, images_to_tensor
from .conversions import tensor_to_bounding_boxes
from .neural_network_classifier import NeuralNetworkClassifier
from .loss_functions import object_detector_loss


class ObjectDetectorModule(torch.nn.Module):

    @classmethod
    def from_dump(cls, dump):
        assert cls.__name__ == dump["type"]
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.classes = dump["classes"]
        obj.cell_size = dump["cell size"]
        obj.boxes_per_cell = dump["boxes per cell"]
        obj.input_norm = BatchNorm2d.from_dump(dump["input norm"])
        obj.encoder = Encoder2d.from_dump(dump["encoder"])
        obj.dense = Dense2d.from_dump(dump["dense"])
        obj.output = Conv2d.from_dump(dump["output"])
        return obj

    def __init__(self, in_channels: int,
                 classes: List[str],
                 boxes_per_cell: int,
                 downsampling: Union[List[dict], List[List[dict]]],
                 pooling: List[Tuple[int, int]],
                 dense: List[dict],
                 pooling_type: str = "max",
                 padded: bool = True,
                 activation: str = "relu",
                 stacked: bool = False,
                 dropout: Union[float, None] = None):
        """
        Parameters
        ----------
        in_channels : int
            the number of channels in the input images
        classes : list of str
            the unique classes the model can predict
        boxes_per_cell : int
            the number of bounding boxe to predict for each grid cell
        downsampling : list of [dict / list of dict]
            the kwargs for the 'Activated2d' layers for all 'downsampling'
        pooling : list of [int / tuple of int]
            the pooling window of all downsampling layers
        dense : list of dict
            the kwargs for the 'Activated2d' of the final 'Dense2d' layer
        pooling_type : one of {'max', 'avg'}
            the type of pooling
        padded : bool
            the default value for the 'padded' key of the kwargs
        activation : str
            the default value for the 'activation' key of the kwargs
        stacked : bool
            the default value for the 'stacked' key of the kwargs
        dropout : float or None
            the default value for the 'dropout' key of the kwargs
        """
        super().__init__()
        assert len(pooling) == len(downsampling)
        self.boxes_per_cell = boxes_per_cell
        self.classes = list(classes)
        self.input_norm = BatchNorm2d(in_channels)
        self.encoder = Encoder2d(in_channels, downsampling, pooling,
                                 pooling_type=pooling_type,
                                 padded=padded,
                                 activation=activation,
                                 stacked=stacked,
                                 dropout=dropout)
        self.cell_size = self.encoder.shape_in([1, 1])
        in_channels = self.encoder.out_channels(in_channels)
        self.dense = Dense2d(in_channels, dense, activation=activation,
                             stacked=stacked, dropout=dropout)
        in_channels = self.dense.out_channels(in_channels)
        out_channels = boxes_per_cell * (5+len(classes))
        self.output = Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        The output of this module is particular.
        It returns the tensors
        (boxe_coords, boxe_size, object_proba, class_proba)
        * 'boxe_coords' is the [x, y] of the predicted boxes in the cell
        * 'boxe_size' is the [width, height] of the predicted boxe
        * 'object_proba' is the probability that an object was found
        * 'class proba' is the probability to be of each class
        Each one is of shape (N, B, C, H, W) with
        * N the number of input images
        * B the number of bounding boxe predicted per grid cell
        * C the number of channels of the tensor (depend on the tensor)
        * H the height of the cell grid
        * W the width of the cell grid

        Parameters
        ----------
        X : torch.Tensor
            The images to process

        Returns
        -------
        tuple of torch.Tensor :
            the (boxe_size, object_proba, class_proba) tensors
        """
        X = self.input_norm(X)
        X = self.encoder(X)
        X = self.dense(X)
        X = self.output(X)
        N, C, H, W = X.shape
        X = X.view(N, self.boxes_per_cell, C//self.boxes_per_cell, H, W)
        boxe_coords = torch.sigmoid(X[:, :, :2, ...])
        boxe_size = X[:, :, 2:4, ...]
        object_proba = torch.sigmoid(X[:, :, 4, ...])
        class_proba = X[:, :, 5:, ...]
        return self._highest_confidence_boxe((boxe_coords, boxe_size,
                                             object_proba, class_proba))

    def loss(self, y_pred: Tuple[torch.Tensor],
             y_target: Tuple[torch.Tensor],
             weights: Union[None, torch.Tensor] = None):
        return object_detector_loss(y_pred, y_target, weights,
                                    self.class_weights)

    def _highest_confidence_boxe(self, tensors: Tuple[torch.Tensor]):
        """
        Returns the predicted boxe with highest confidence for each anchor cell
        """
        boxe_coords, boxe_size, object_proba, class_proba = tensors
        object_proba, indexes = torch.max(object_proba, dim=1)
        indexes = indexes.unsqueeze(1)
        boxes_index = indexes.expand((-1, 2, -1, -1)).unsqueeze(1)
        boxe_coords = torch.gather(boxe_coords, 1, boxes_index).squeeze(1)
        boxe_size = torch.gather(boxe_size, 1, boxes_index).squeeze(1)
        _, _, n, _, _ = class_proba.shape
        class_proba = torch.gather(class_proba, 1,
                                   indexes.expand((-1, n, -1, -1)).unsqueeze(1)
                                   ).squeeze(1)
        return boxe_coords, boxe_size, object_proba, class_proba

    @property
    def dump(self):
        return {"type": type(self).__name__,
                "classes": list(self.classes),
                "cell size": list(self.cell_size),
                "boxes per cell": self.boxes_per_cell,
                "input norm": self.input_norm.dump,
                "encoder": self.encoder.dump,
                "dense": self.dense.dump,
                "output": self.output.dump}


class ObjectDetector(NeuralNetworkClassifier):

    ModuleType = ObjectDetectorModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _data_to_tensor(self, X: np.ndarray,
                        Y: Union[None, List[dict]],
                        weights: None = None,
                        device: torch.device = torch.device("cpu")) -> tuple:
        if weights is not None:
            raise ValueError("The weight of each observation must be passed in"
                             " the dictionary of bounding boxes, not as a"
                             " separate list")
        x = images_to_tensor(X, device)
        if Y is not None:
            r = bounding_boxes_to_tensor(Y, tuple(X.shape[1:3]),
                                         self.module.cell_size,
                                         self.module.classes, device)
            boxe_coords, boxe_size, object_mask, class_index, cell_weights = r
            y = boxe_coords, boxe_size, object_mask, class_index
            w = cell_weights
        else:
            y = None
            w = None
        return x, y, w

    def _tensor_to_y(self, tensors: Tuple[torch.Tensor]) -> List[dict]:
        return tensor_to_bounding_boxes(tensors, self.module.cell_size,
                                        self.module.classes)
