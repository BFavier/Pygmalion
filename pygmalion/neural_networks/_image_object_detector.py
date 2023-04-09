import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from typing import List, Sequence, Iterable, Tuple, Optional
from .layers import ConvBlock
from ._conversions import tensor_to_classes
from ._conversions import classes_to_tensor, images_to_tensor, floats_to_tensor
from ._conversions import tensor_to_probabilities
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy, MSE


class ImageObjectDetector(NeuralNetworkClassifier):

    def __init__(self, in_channels: int,
                 classes: Iterable[str],
                 features: Iterable[int],
                 kernel_size: Tuple[int, int] = (3, 3),
                 pooling_size: Optional[Tuple[int, int]] = (2, 2),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
                 n_convs_per_block: int = 1,
                 normalize: bool = True,
                 residuals: bool = True,
                 dropout: Optional[float] = None):
        """
        Parameters
        ----------
        ...
        """
        super().__init__(classes)
        self.downscaling_factor = tuple((s*mp) for s, mp in zip(stride, pooling_size or (1, 1)))
        self.n_stages = len(features)
        self.layers = torch.nn.ModuleList()
        in_features = in_channels
        for out_features in features:
            self.layers.append(
                ConvBlock(in_features, out_features, kernel_size, stride, activation,
                          normalize, residuals, n_convs_per_block, dropout))
            if pooling_size is not None:
                self.layers.append(torch.nn.MaxPool2d(pooling_size))
            in_features = out_features
        self.detected = torch.nn.Conv2d(out_features, 1, (1, 1))
        self.positions = torch.nn.Conv2d(out_features, 2, (1, 1))
        self.dimensions = torch.nn.Conv2d(out_features, 2, (1, 1))
        self.objects_class = torch.nn.Conv2d(out_features, len(self.classes), (1, 1))

    def forward(self, X: torch.Tensor):
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of float images of shape (N, C, H, W)
        
        Returns
        -------
        tuple of torch.Tensor :
            returns (detected, position, dimension, object_class) with
            * 'detected' object detection probability in each cell,
              tensor of floats of shape (N, h, w)
            * 'position' relative (x, y) position of the detected object's center in each cell,
               tensor of floats of shape (N, 2, h, w)
            * 'dimension' relative (w, h) dimension of the detected object in each cell,
              tensor of floats of shape (N, 2, h, w)
            * 'object_class' probability (once softmaxed) of each class in each cell,
              tensor of floats of shape (N, n_classes, h, w)
        """
        X = X.to(self.device)
        for layer in self.layers:
            X = layer(X)
        N, C, H, W = X.shape
        detected = torch.sigmoid(self.detected(X).squeeze(1))
        position = torch.sigmoid(self.positions(X))
        dimension = torch.log(1 + torch.exp(self.dimensions(X)))
        object_class = self.objects_class(X)
        return detected, position, dimension, object_class

    def loss(self, x: torch.Tensor, detected: torch.Tensor,
             positions: torch.Tensor, dimensions: torch.Tensor,
             object_class: torch.Tensor,
             weights: Optional[torch.Tensor] = None,
             class_weights: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of float images of shape (N, C, H, W)
        detected : torch.Tensor
            tensor of booleans of shape (N, h, w) indicating presence of an object in a cell
        positions : torch.Tensor
            tensor of floats of shape (N, 2, h, w) of (x, y) position of detected objects in the cells
        dimension : torch.Tensor
            tensor of floats of shape (N, 2, h, w) of (width, height) of detected objects in each cell
        object_class : torch.Tensor
            tensor of longs of shape (N, h, w) of target class for each cell
        """
        detected_pred, position_pred, dimension_pred, class_pred = self(x)
        presence_loss = F.binary_cross_entropy(detected_pred, detected.float().to(detected_pred.device), weight=weights)
        if detected.any():
            p_subset = detected.unsqueeze(1).expand(-1, 2, -1, -1)
            return (presence_loss
                    + MSE(position_pred[p_subset], positions[p_subset], weights)
                    + MSE(dimension_pred[p_subset], dimensions[p_subset], weights)
                    + cross_entropy(class_pred.permute(0, 2, 3, 1)[detected], object_class[detected], weights=weights, class_weights=class_weights))
        else:
            return presence_loss
    
    def predict(self, images: np.ndarray, detection_treshold: float=0.5,
                threshold_intersect: float = 0.8,
                downscaling_factors: List[int] = [1]) -> List[dict]:
        """
        """
        n = len(images)
        predictions = [{"x": [], "y": [], "w": [], "h": [], "class": [],
                        "detection confidence": [], "class confidence": []}
                       for _ in range(n)]
        self.eval()
        X = self._x_to_tensor(images, self.device)
        for df in downscaling_factors:
            with torch.no_grad():
                detected, position, dimension, object_class = self(F.avg_pool2d(X, kernel_size=(df, df)))
            h_image, w_image = images.shape[1:3]
            h_cell, w_cell = (f**self.n_stages for f in self.downscaling_factor)
            # converting from grid coordinates to pixel coordinates
            grid_pos = torch.stack(torch.meshgrid(torch.arange(0, w_image, w_cell, dtype=position.dtype, device=self.device),
                                torch.arange(0, h_image, h_cell, dtype=position.dtype, device=self.device),
                                indexing="xy"), dim=0)
            cell_dimension = torch.tensor([w_cell, h_cell], dtype=torch.float, device=self.device).reshape(1, 2, 1, 1)
            pixel_position = grid_pos.unsqueeze(0) + position * cell_dimension
            pixel_dimension = dimension * cell_dimension
            # selecting cells with detected objects
            subset = detected > detection_treshold
            probabilities, classes = torch.softmax(object_class, dim=1).max(dim=1)
            for i, (sub, det, pos, dim, prob, cls) in enumerate(zip(subset, detected, pixel_position, pixel_dimension, probabilities, classes)):
                det = det[sub]
                pos = pos.permute(1, 2, 0)[sub]
                dim = dim.permute(1, 2, 0)[sub]
                prob = prob[sub]
                cls = cls[sub]
                predictions[i]["x"].extend(pos[:, 0].cpu().tolist())
                predictions[i]["y"].extend(pos[:, 1].cpu().tolist())
                predictions[i]["w"].extend(dim[:, 0].cpu().tolist())
                predictions[i]["h"].extend(dim[:, 1].cpu().tolist())
                predictions[i]["class"].extend([self.classes[i] for i in cls.cpu().tolist()])
                predictions[i]["detection confidence"].extend(det.cpu().tolist())
                predictions[i]["class confidence"].extend(prob.cpu().tolist())
        # applying non max suppression
        pass
        return predictions

    # def _non_max_suppression(self, bboxes: dict, threshold_intersect: float) -> dict:
    #     """
    #     Perform non max suppression
    #     """
    #     x, y, w, h, c = (bboxes[c] for c in ("x", "y", "w", "h", "class"))

    @property
    def device(self) -> torch.device:
        return self.detected.weight.device

    def _x_to_tensor(self, x: np.ndarray,
                     device: Optional[torch.device] = None):
        return images_to_tensor(x, device=device)

    def _y_to_tensor(self, x: np.ndarray, y: Iterable[dict],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        grid_h, grid_w = (f ** self.n_stages for f in self.downscaling_factor)
        n, h, w = x.shape[:3]
        h, w = (h//grid_h, w//grid_w)
        detected = torch.zeros((n, h, w), dtype=torch.bool)
        positions = torch.zeros((n, 2, h, w), dtype=torch.float)
        dimensions = torch.zeros((n, 2, h, w), dtype=torch.float)
        classes = torch.zeros((n, h, w), dtype=torch.long)
        for n, bboxes in enumerate(y):
            X, Y, W, H = (floats_to_tensor(bboxes[v]) for v in ("x", "y", "w", "h"))
            C = classes_to_tensor(bboxes["class"], self.classes, device=device)
            i, j = (torch.div(Y, grid_h, rounding_mode="floor").long(), torch.div(X, grid_w, rounding_mode="floor").long())
            py, px = ((Y % grid_h) / grid_h, (X % grid_w) / grid_w)
            detected[n, i, j] = 1
            positions[n, :, i, j] = torch.stack([px, py], dim=0)
            dimensions[n, :, i, j] = torch.stack([W / grid_w, H / grid_h], dim=0)
            classes[n, i, j] = C
        return detected, positions, dimensions, classes

    def data_to_tensor(self, x: np.ndarray, y: List[dict],
                       weights: Optional[Sequence[float]] = None,
                       class_weights: Optional[Sequence[float]] = None,
                       device: Optional[torch.device] = None, **kwargs) -> tuple:
        """
        """
        images = self._x_to_tensor(x, device, **kwargs)
        detected, positions, dimensions, classes = self._y_to_tensor(x, y, device, **kwargs)
        return images, detected, positions, dimensions, classes

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_classes(tensor, self.classes)

    def _tensor_to_proba(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_probabilities(tensor, self.classes)
