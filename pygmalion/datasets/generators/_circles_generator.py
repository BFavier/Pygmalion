from typing import Tuple
import torch
import numpy as np


class CirclesGenerator:
    """
    A generator that generates on the fly batches of (image, segmentation)
    with 'image' grayscale images of circles boundaries
    and 'segmentation' the binary image of circle interiors
    """

    def __init__(self, batch_size: int, n_batches: int,
                 image_size: Tuple[int, int] = (64, 64),
                 n_max_circles: int = 5, thickness: float = 1.,
                 radius_fraction: Tuple[float, float] = (0.05, 0.2),
                 device: torch.device = torch.device("cpu")):
        """
        Parameters
        ----------
        batch_size : int
            the number of images in a batch
        n_batches : int
            the number of batches yielded per iteration
        image_size : tuple of (int, int)
            the (height, width) of the generated images
        n_max_circles : int
            the number of circles generated in each image is between 1 and 'n_max_circles'
        thickness : int
            the radius in pixels of the circle boundary
        radius_fraction : tuple of (float, float)
            the (low, high) range for the radius of each circle
            these values are multiplied by min(height, width)
        """
        self.batche_size = batch_size
        self.n_batches = n_batches
        self.height, self.width = image_size
        self.n_max_circles = n_max_circles
        self.thickness = thickness
        self.radius_fraction = radius_fraction
        self.device = device

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self._generate_circles(self.batche_size, self.device)
    
    def generate(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        generate a tuple of (input_images, target_segmentations)
        """
        image, target = self._generate_circles(n, torch.device("cpu"))
        image = (255 * image.squeeze(1).numpy()).astype(np.uint8)
        target = target.numpy()
        return image, target

    def _generate_circles(self, n_images: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        generate a batch of (image, binary) tensors for training
        with 'image' a tensor of shape (N, 1, H, W) containing grayscale images of circles
        and 'binary' a tensor of long of shape (N, H, W) containing 1 if inside of a circle and 0 otherwise
        """
        low, high = sorted(int(round(rf*min(self.height, self.width))) for rf in self.radius_fraction)
        r = torch.rand(size=(n_images, self.n_max_circles, 1, 1), device=device) * (high - low) + low
        x = torch.rand(size=(n_images, self.n_max_circles, 1, 1), device=device) * self.width
        y = torch.rand(size=(n_images, self.n_max_circles, 1, 1), device=device) * self.height
        Y, X = (v.reshape(1, 1, self.height, self.width) for v in
                torch.meshgrid([torch.arange(self.height, device=device),
                                torch.arange(self.width, device=device)],
                               indexing="ij"))
        Y, X = Y+0.5, X+0.5
        selected = (torch.arange(self.n_max_circles, device=device).reshape(1, self.n_max_circles, 1, 1)
                    <= torch.randint(1, self.n_max_circles+1, size=(n_images, 1, 1, 1), device=device))
        distances = torch.sqrt((Y - y)**2 + (X - x)**2)
        boundaries = ((torch.abs(distances - r) < self.thickness/2) & selected).any(dim=1).unsqueeze(1).float()
        filled = ((distances <= r + self.thickness/2) & selected).any(dim=1).long()
        return boundaries, filled
