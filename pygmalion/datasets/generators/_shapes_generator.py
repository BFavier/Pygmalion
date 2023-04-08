from typing import Tuple, List
import numpy as np


class ShapesGenerator:
    """
    A generator that generates on the fly batches of (images, bboxes)
    with 'images' grayscale images of circles and squares
    and 'bboxes' a list of dictionnary describing the objects found in each image
    """

    def __init__(self, batch_size: int, n_batches: int,
                 image_size: Tuple[int, int] = (64, 64),
                 n_max_shapes: int = 3,
                 radius_fraction: Tuple[float, float] = (0.05, 0.2)):
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
        radius_fraction : tuple of (float, float)
            the (low, high) range for the radius of each circle
            these values are multiplied by min(height, width)
        """
        self.batche_size = batch_size
        self.n_batches = n_batches
        self.height, self.width = image_size
        self.n_max_shapes = n_max_shapes
        self.radius_fraction = radius_fraction

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.generate(self.batche_size)

    def generate(self, n_images: int) -> Tuple[np.ndarray, List[dict]]:
        """
        generate a batch of (image, bboxes) tensors for training
        with 'image' an array of shape (N, H, W) containing grayscale images of circles
        and 'bboxes' a list of dict describing the bounding boxes contained in each image
        """
        low, high = sorted(int(round(rf*min(self.height, self.width))) for rf in self.radius_fraction)
        r = np.random.rand(n_images, self.n_max_shapes, 1, 1) * (high - low) + low
        n_pos = int(1/max(self.radius_fraction))
        pos = np.arange(n_pos**2)
        pos_idx = np.random.rand(n_images, n_pos**2).argsort(axis=1)[:, :self.n_max_shapes]
        x = (pos[pos_idx] % n_pos + 0.5).reshape(n_images, self.n_max_shapes, 1, 1) / n_pos * self.width + 0.5
        y = (pos[pos_idx] // n_pos + 0.5).reshape(n_images, self.n_max_shapes, 1, 1) / n_pos * self.height + 0.5
        X, Y = (v.reshape(1, 1, self.height, self.width) for v in
                np.meshgrid(np.arange(self.width), np.arange(self.height)))
        Y, X = Y+0.5, X+0.5
        selected = (np.arange(self.n_max_shapes).reshape(1, self.n_max_shapes, 1, 1)
                    <= np.random.randint(1, self.n_max_shapes+1, size=(n_images, 1, 1, 1)))
        is_circle = (np.random.rand(n_images, self.n_max_shapes) < 0.5).reshape(n_images, self.n_max_shapes, 1, 1)
        distances = np.sqrt((Y - y)**2 + (X - x)**2)
        images = np.any(((((distances <= r) & is_circle) |
                         ((np.abs(X-x) < r) & (np.abs(Y-y) < r) & ~is_circle)
                         ) & selected), axis=1)
        images = (255 * images.astype(np.uint8).reshape(n_images, self.height, self.width))
        bboxes = [{"x": [v for u, v in zip(_s.reshape(-1), _x.reshape(-1)) if u],
                   "y": [v for u, v in zip(_s.reshape(-1), _y.reshape(-1)) if u],
                   "w": [2*v for u, v in zip(_s.reshape(-1), _r.reshape(-1)) if u],
                   "h": [2*v for u, v in zip(_s.reshape(-1), _r.reshape(-1)) if u],
                   "class": ["circle" if v else "square" for u, v in zip(_s.reshape(-1), _c.reshape(-1)) if u]}
                  for _s, _x, _y, _r, _c in zip(selected, x, y, r, is_circle)]
        return images, bboxes
