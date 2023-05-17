import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from typing import List, Sequence, Iterable, Tuple, Optional
from itertools import count
from .layers.convolutions import ConvolutionalEncoder, PaddedConv2d
from ._conversions import tensor_to_classes
from ._conversions import classes_to_tensor, images_to_tensor, floats_to_tensor
from ._conversions import tensor_to_probabilities
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy, RMSE


class ImageObjectDetector(NeuralNetworkClassifier):

    def __init__(self, in_channels: int,
                 classes: Iterable[str],
                 features: Iterable[int],
                 bboxes_per_cell: int = 5,
                 kernel_size: Tuple[int, int] = (3, 3),
                 pooling_size: Tuple[int, int] = (2, 2),
                 stride: Tuple[int, int] = (1, 1),
                 activation: str = "relu",
                 n_convs_per_block: int = 1,
                 normalize: bool = True,
                 residuals: bool = True,
                 dropout: Optional[float] = None,
                 low_memory: bool = True):
        """
        Parameters
        ----------
        ...
        """
        super().__init__(classes)
        self.downsampling_window = tuple((s*mp) for s, mp in zip(stride or (1, 1), pooling_size or (1, 1)))
        self.cells_dimensions = tuple(dw ** len(features) for dw in self.downsampling_window)
        self.layers = torch.nn.ModuleList()
        self.bboxes_per_cell = bboxes_per_cell
        self.encoder = ConvolutionalEncoder(
            in_channels, features, kernel_size, pooling_size, stride, activation,
            n_convs_per_block, normalize, residuals, dropout, low_memory)
        self.confidence = PaddedConv2d(features[-1], self.bboxes_per_cell, kernel_size)
        self.positions = PaddedConv2d(features[-1], self.bboxes_per_cell*2, kernel_size)
        self.dimensions = PaddedConv2d(features[-1], self.bboxes_per_cell*2, kernel_size)
        self.objects_class = PaddedConv2d(features[-1], self.bboxes_per_cell*len(self.classes), kernel_size)

    def forward(self, X: torch.Tensor):
        """
        Parameters
        ----------
        X : torch.Tensor
            tensor of float images of shape (N, C, H, W)
        
        Returns
        -------
        tuple of torch.Tensor :
            returns (confidence, position, dimension, object_class) with
            * 'confidence' predicted IoU of bounding boxe (once sigmoid is applied) in each cell,
              tensor of floats of shape (N, bboxes_per_cell, h, w)
            * 'position' relative (x, y) position of the detected object's center in each cell,
               tensor of floats of shape (N, bboxes_per_cell, 2, h, w)
            * 'dimension' relative (w, h) dimension of the detected object in each cell,
              tensor of floats of shape (N, bboxes_per_cell, 2, h, w)
            * 'object_class' probability (once softmaxed) of each class in each cell,
              tensor of floats of shape (N, bboxes_per_cell, n_classes, h, w)
        """
        X = X.to(self.device)
        X = self.encoder(X)
        N, C, H, W = X.shape
        confidence = torch.sigmoid(self.confidence(X)).reshape(N, self.bboxes_per_cell, H, W)
        position = torch.sigmoid(self.positions(X)).reshape(N, self.bboxes_per_cell, 2, H, W)
        dimension = torch.log(1 + torch.exp(self.dimensions(X))).reshape(N, self.bboxes_per_cell, 2, H, W)
        object_class = self.objects_class(X).reshape(N, self.bboxes_per_cell, len(self.classes), H, W)
        return confidence, position, dimension, object_class

    def _weighted_mean(self, tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        """
        return (tensor * weights).sum() / weights.sum()

    def loss(self, x: torch.Tensor, presence: torch.Tensor,
             positions: torch.Tensor, dimensions: torch.Tensor,
             object_class: torch.Tensor,
             weights: Optional[torch.Tensor] = None,
             class_weights: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of float images of shape (N, C, H, W)
        presence : torch.Tensor
            tensor of booleans of shape (N, h, w) indicating presence of an object in a cell
        positions : torch.Tensor
            tensor of floats of shape (N, 2, h, w) of (x, y) position of objects in the cells
        dimension : torch.Tensor
            tensor of floats of shape (N, 2, h, w) of (width, height) of objects in each cell
        object_class : torch.Tensor
            tensor of longs of shape (N, h, w) of target class for each cell
        """
        presence, positions, dimensions, object_class = (t.to(self.device) for t in (presence, positions, dimensions, object_class))
        confidence_pred, position_pred, dimension_pred, class_pred = self(x)
        N, H, W = presence.shape
        # Calculate IoU
        IoU = self._intersect_over_union(
            position_pred[:, :, 0, ...], position_pred[:, :, 1, ...], dimension_pred[:, :, 0, ...], dimension_pred[:, :, 1, ...],
            positions[:, 0, ...].unsqueeze(1), positions[:, 1, ...].unsqueeze(1), dimensions[:, 0, ...].unsqueeze(1), dimensions[:, 1, ...].unsqueeze(1)
            ).squeeze(2).permute(0, 2, 3, 1)  # IoU of shape (N, H, W, bboxes_per_cell)
        # changing axes order and calculating weights
        confidence_pred, position_pred, dimension_pred, class_pred = (
            torch.moveaxis(t, 1, -1) for t in (confidence_pred, position_pred, dimension_pred, class_pred))  # reshape to (N, *, H, W, bboxes_per_cell)
        weights = self._softscale(confidence_pred, dim=-1)  # weights of shape (N, H, W, bboxes_per_cell)
        # Compute losses
        absent = ~presence
        absence_loss = self._weighted_mean(-torch.log(1 - confidence_pred[absent]), weights[absent])
        if presence.any():
            presence_2d = presence.unsqueeze(1).expand(-1, 2, -1, -1)
            weights_presence_2d = weights.unsqueeze(1).expand(-1, 2, -1, -1, -1)[presence_2d]
            bboxe_confidence_loss = RMSE(confidence_pred[presence], IoU[presence])
            return (absence_loss + bboxe_confidence_loss 
                    + RMSE(position_pred[presence_2d], positions[presence_2d].unsqueeze(-1), weights=weights_presence_2d)
                    + RMSE(dimension_pred[presence_2d], dimensions[presence_2d].unsqueeze(-1), weights=weights_presence_2d)
                    + cross_entropy(class_pred.moveaxis(1, -2)[presence],
                                    object_class[presence].unsqueeze(-1).expand(-1, self.bboxes_per_cell),
                                    weights=weights[presence],
                                    class_weights=class_weights))
        else:
            return absence_loss

    def predict(self, images: np.ndarray, detection_treshold: float=0.5,
                threshold_intersect: Optional[float] = 0.6,
                multi_scale: bool = False) -> List[dict]:
        """
        """
        n, h_image, w_image = images.shape[:3]
        predictions = [{"x": [], "y": [], "w": [], "h": [], "class": [],
                        "bboxe confidence": [], "class confidence": []}
                       for _ in range(n)]
        self.eval()
        X = self._x_to_tensor(images, self.device)
        for i in count():
            h_down, w_down = tuple(s**i for s in self.downsampling_window)
            if any(s // (d*g) == 0 for s, d, g in zip((h_image, w_image), (h_down, w_down), self.cells_dimensions)):
                break
            with torch.no_grad():
                confidence, position, dimension, object_class = self(F.avg_pool2d(X, kernel_size=(h_down, w_down)))
            h_cell, w_cell = self.cells_dimensions
            N, _, h_grid, w_grid = confidence.shape
            # select most confident bboxe for each cell
            confidence, bboxe_index = confidence.max(dim=1)
            position, dimension, object_class = (
                torch.gather(tensor, 1, bboxe_index.reshape(N, 1, 1, h_grid, w_grid).expand(-1, -1, tensor.shape[2], -1, -1)).squeeze(1)
                for tensor in (position, dimension, object_class))
            # converting from grid coordinates to pixel coordinates
            grid_pos = torch.stack(torch.meshgrid(torch.arange(0, w_image, w_cell*w_down, dtype=position.dtype, device=self.device),
                                                  torch.arange(0, h_image, h_cell*h_down, dtype=position.dtype, device=self.device),
                                                  indexing="xy"),
                                   dim=0)
            cell_dimension = torch.tensor([w_cell, h_cell], dtype=torch.float, device=self.device).reshape(1, 2, 1, 1)
            pixel_position = grid_pos.unsqueeze(0) + position * cell_dimension
            pixel_dimension = dimension * cell_dimension
            # selecting cells with detected objects
            subset = confidence > detection_treshold
            probabilities, classes = torch.softmax(object_class, dim=1).max(dim=1)
            for i, (sub, conf, pos, dim, prob, cls) in enumerate(zip(subset, confidence, pixel_position, pixel_dimension, probabilities, classes)):
                conf = conf[sub]
                pos = pos.permute(1, 2, 0)[sub]
                dim = dim.permute(1, 2, 0)[sub]
                prob = prob[sub]
                cls = cls[sub]
                predictions[i]["x"].extend(pos[:, 0].cpu().tolist())
                predictions[i]["y"].extend(pos[:, 1].cpu().tolist())
                predictions[i]["w"].extend(dim[:, 0].cpu().tolist())
                predictions[i]["h"].extend(dim[:, 1].cpu().tolist())
                predictions[i]["class"].extend([self.classes[i] for i in cls.cpu().tolist()])
                predictions[i]["bboxe confidence"].extend(conf.cpu().tolist())
                predictions[i]["class confidence"].extend(prob.cpu().tolist())
            if not multi_scale:
                break
        # applying non max suppression
        if threshold_intersect is not None:
            predictions = [self._non_max_suppression(bboxes, threshold_intersect) for bboxes in predictions]
        return predictions

    @staticmethod
    def _non_max_suppression(bboxes: dict, threshold_intersect: float) -> dict:
        """
        Perform non max suppression
        """
        # read and sort bboxes by increasing confidence
        x, y, w, h, b = (bboxes[c] for c in ("x", "y", "w", "h", "bboxe confidence"))
        indexes = [i for i, v in sorted(enumerate(b), key=lambda x: x[1])]
        x, y, w, h = (torch.tensor(v).unsqueeze(0) for v in (x, y, w, h))
        # compute cross intersects over union
        IoU = ImageObjectDetector._intersect_over_union(x, y, w, h,
                                                        x, y, w, h)
        # select bboxes as long as they dont intersect too much with already selected bboxes
        remaining = indexes
        selected = []
        while len(remaining) > 0:
            i = remaining.pop()
            if len(selected) == 0 or (IoU[0, i, selected] <= threshold_intersect).all():
                selected.append(i)
        return {k: [v[i] for i in selected] for k, v in bboxes.items()}

    @staticmethod
    def _intersect_over_union(Ax: torch.Tensor, Ay: torch.Tensor, Aw: torch.Tensor, Ah: torch.Tensor,
                              Bx: torch.Tensor, By: torch.Tensor, Bw: torch.Tensor, Bh: torch.Tensor,
                              eps: float = 1.0E-10) -> torch.Tensor:
        """
        Compute cross table of intersect area between N two sets of rectangles A and B.

        Parameters
        ----------
        Ax : torch.Tensor
            x position of the center of A. Tensor of floats of shape (N, nA, *)
        Ay : torch.Tensor
            y position of the center of A. Tensor of floats of shape (N, nA, *)
        Aw : torch.Tensor
            Width of A. Tensor of floats of shape (N, nA, *)
        Ah : torch.Tensor
            Height of A. Tensor of floats of shape (N, nA, *)
        Bx : torch.Tensor
            x position of the center of B. Tensor of floats of shape (N, nB, *)
        By : torch.Tensor
            y position of the center of B. Tensor of floats of shape (N, nB, *)
        Bw : torch.Tensor
            Width of B. Tensor of floats of shape (N, nB, *)
        Bh : torch.Tensor
            Height of B. Tensor of floats of shape (N, nB, *)
        eps : float
            epsilon to avoid division by zero
        
        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, Na, Nb, *)
        """
        Ax1, Ay1, Ax2, Ay2 = Ax-Aw/2, Ay-Ah/2, Ax+Aw/2, Ay+Ah/2
        Bx1, By1, Bx2, By2 = Bx-Bw/2, By-Bh/2, Bx+Bw/2, By+Bh/2
        # coordinates of intersect rectangles, tensors of shape (N, Na, Nb, *)
        y_top = torch.maximum(Ay1.unsqueeze(2), By1.unsqueeze(1))
        y_bot = torch.minimum(Ay2.unsqueeze(2), By2.unsqueeze(1))
        x_left = torch.maximum(Ax1.unsqueeze(2), Bx1.unsqueeze(1))
        x_right = torch.minimum(Ax2.unsqueeze(2), Bx2.unsqueeze(1))
        # calculating intersect area
        intersect = (y_bot-y_top) * (x_right-x_left) * ((x_left < x_right) & (y_top < y_bot))
        # calculating union
        union = (Ah * Aw).unsqueeze(2) + (Bh * Bw).unsqueeze(1) - intersect
        return intersect / (union + eps)
    
    @staticmethod
    def _softscale(X: torch.Tensor, dim: int=0) -> torch.Tensor:
        X = torch.exp(X)
        return X / X.max(dim=dim).values.unsqueeze(dim)

    @property
    def device(self) -> torch.device:
        return self.confidence.conv.weight.device

    def _x_to_tensor(self, x: np.ndarray,
                     device: Optional[torch.device] = None):
        return images_to_tensor(x, device=device)

    def _y_to_tensor(self, y: Iterable[dict], image_h: int, image_w: int,
                     device: Optional[torch.device] = None) -> torch.Tensor:
        n = len(y)
        cell_h, cell_w = self.cells_dimensions
        grid_h, grid_w = (image_h//cell_h, image_w//cell_w)
        presence = torch.zeros((n, grid_h, grid_w), dtype=torch.bool)
        positions = torch.zeros((n, 2, grid_h, grid_w), dtype=torch.float)
        dimensions = torch.zeros((n, 2, grid_h, grid_w), dtype=torch.float)
        classes = torch.zeros((n, grid_h, grid_w), dtype=torch.long)
        for n, bboxes in enumerate(y):
            X, Y, W, H = (floats_to_tensor(bboxes[v]) for v in ("x", "y", "w", "h"))
            C = classes_to_tensor(bboxes["class"], self.classes, device=device)
            i, j = (torch.div(Y, cell_h, rounding_mode="floor").long(), torch.div(X, cell_w, rounding_mode="floor").long())
            py, px = ((Y % cell_h) / cell_h, (X % cell_w) / cell_w)
            presence[n, i, j] = 1
            positions[n, :, i, j] = torch.stack([px, py], dim=0)
            dimensions[n, :, i, j] = torch.stack([W / cell_w, H / cell_h], dim=0)
            classes[n, i, j] = C
        return presence, positions, dimensions, classes

    def data_to_tensor(self, x: np.ndarray, y: List[dict],
                       class_weights: Optional[Sequence[float]] = None,
                       device: Optional[torch.device] = None, **kwargs) -> tuple:
        """
        """
        images = self._x_to_tensor(x, device, **kwargs)
        h, w = x.shape[1:3]
        presence, positions, dimensions, classes = self._y_to_tensor(y, h, w, device, **kwargs)
        return images, presence, positions, dimensions, classes

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_classes(tensor, self.classes)

    def _tensor_to_proba(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_probabilities(tensor, self.classes)
