import pathlib
import torch
import torch.nn.functional as F
import numpy as np
import pygmalion as ml
import matplotlib.pyplot as plt
import math
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
ml.datasets.aquarium(data_path)

data = dict(np.load(data_path / "aquarium.npz"))
images = data["images"]
image_indexes = data["image_indexes"]
classes = [c.decode("utf-8") for c in data["class_names"].tolist()]
bboxes = [{"class": [classes[i] for i in data["class_indexes"][image_indexes == i]],
            **{k: data[k][image_indexes == i] for k in ("x", "y", "w", "h")}}
           for i in range(len(images))]

model = ml.neural_networks.ImageObjectDetector(3, classes, [8, 16, 32, 64, 128], n_convs_per_block=1, dropout=None)


class Batchifyer:

    def __init__(self, images: np.ndarray, bboxes: list, batch_size: int, n_batches: int,
                 data_augmentation: bool = True, device: torch.device = "cpu",
                 resolutions: list[float] = [1.0], max_rotation_angle: float = 5.0,
                 max_downscaling_factor: float =0.90):
        self.images, self.bboxes = model._x_to_tensor(images), bboxes
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.device = device
        self.data_augmentation = data_augmentation
        self.resolutions = resolutions
        self.max_rotation_angle = max_rotation_angle
        self.max_downscaling_factor = max_downscaling_factor

    def __iter__(self):
        shuffle = np.random.permutation(len(self.images))
        _, _, h, w = self.images.shape
        cell_h, cell_w = model.cells_dimensions
        for i in range(self.n_batches):
            idx = shuffle[i*self.batch_size:(i+1)*self.batch_size]
            X, Y = self.images.to(self.device)[idx], [self.bboxes[i] for i in idx]
            for resolution in self.resolutions:
                if self.data_augmentation:
                    n = len(idx)
                    max_theta = self.max_rotation_angle * (math.pi / 180.0)
                    wh = torch.tensor([w, h], dtype=torch.float32, device=self.device).reshape(1, 1, 1, 2)
                    grid = torch.stack(torch.meshgrid([torch.linspace(-w/2, w/2, int(w*resolution), device=self.device),
                                                    torch.linspace(-h/2, h/2, int(h*resolution), device=self.device)],
                                    indexing="xy"), dim=-1)
                    # random horizontal flips
                    one = torch.ones(n, device=self.device)
                    factor = torch.stack([torch.where(torch.rand(n, device=self.device) > 0.5, one, -one), one], dim=-1)
                    grid = grid.reshape(1, h, w, 2) * factor.reshape(n, 1, 1, 2)
                    # random rotations
                    theta = torch.rand(n, device=self.device) * (2*max_theta) -  max_theta
                    sin, cos = torch.sin(theta), torch.cos(theta)
                    rot = torch.stack([torch.stack([cos, -sin], dim=-1),
                                    torch.stack([sin, cos], dim=-1)], dim=-2)
                    grid = (rot.reshape(n, 1, 1, 2, 2) @ grid.reshape(n, h, w, 2, 1)).squeeze(-1)
                    # random downscaling
                    scale = (1 - torch.rand(n, device=self.device) * (1 - self.max_downscaling_factor)) / (grid*2/wh).reshape(n, -1).abs().max(dim=1).values
                    grid = grid * scale.reshape(n, 1, 1, 1)
                    # random offset
                    delta = (wh.reshape(1, 2)/2 - grid.reshape(n, -1, 2).max(dim=1).values)
                    offset = torch.rand((n, 2), device=self.device) * (2*delta) - delta
                    grid = grid + offset.reshape(n, 1, 1, 2)
                    # converting back to (-1, 1) range
                    grid = torch.clip(grid*2 / wh, -1, 1)
                    X = F.grid_sample(X, grid, mode="bilinear", align_corners=True, padding_mode="border")
                    # converting bboxe coordinates
                    wh = wh.reshape(1, 2)
                    coords = [tuple(Yb[k] for k in "xywh") for Yb in Y]
                    corners = [torch.stack([torch.stack([torch.tensor(x) - torch.tensor(w)/2,
                                                         torch.tensor(y) - torch.tensor(h)/2], dim=-1)/wh,
                                            torch.stack([torch.tensor(x) + torch.tensor(w)/2,
                                                         torch.tensor(y) - torch.tensor(h)/2], dim=-1)/wh,
                                            torch.stack([torch.tensor(x) + torch.tensor(w)/2,
                                                         torch.tensor(y) + torch.tensor(h)/2], dim=-1)/wh,
                                            torch.stack([torch.tensor(x) - torch.tensor(w)/2,
                                                         torch.tensor(y) + torch.tensor(h)/2], dim=-1)/wh], dim=1)
                               for x, y, w, h in coords]
                    grid = (grid+1)/2
                    M = grid[:, 0, 0, :]
                    ux, uy = (grid[:, 0, -1, :] - M, grid[:, -1, 0, :] - M)
                    corners = [torch.stack([torch.linalg.vecdot(c - m.reshape(1, 1, 2), u.reshape(1, 1, 2), dim=-1)/torch.linalg.norm(u, ord=2)**2
                                            for u in (_ux, _uy)], dim=-1)
                               for c, m, _ux, _uy in zip(corners, M, ux, uy)]
                    # filter bboxe centers outside of image
                    centers = [c.mean(dim=1) for c in corners]
                    keeps = [torch.all((0 <= center) & (center <= 1), dim=-1) for center in centers]
                    corners = [c[k] for c, k in zip(corners, keeps)]
                    classes = [[c for c, k in zip(Yb["class"], keep) if k] for Yb, keep in zip(Y, keeps)]
                    # clip bboxe corners to [0, 1] range
                    corners = [torch.clip(c, -1, 1) for c in corners]
                    # convert to (x, y, w, h) coordinates
                    inf, sup, mean = [c.min(dim=1).values for c in corners], [c.max(dim=1).values for c in corners], [c.mean(dim=1) for c in corners]
                    gh, gw = grid.shape[1:-1]
                    x, y = [m[:, 0]*gw for m in mean], [m[:, 1]*gh for m in mean]
                    w, h = [(s[:, 0] - i[:, 0])*gw for i, s in zip(inf, sup)], [(s[:, 1] - i[:, 1])*gh for i, s in zip(inf, sup)]
                    Y = [{"class": cls, "x": _x, "y": _y, "w": _w, "h": _h}
                         for _x, _y, _w, _h, cls in zip(x, y, w, h, classes)]
                h, w = X.shape[-2:]
                yield X, Y#model._y_to_tensor(Y, h, w, device=self.device)

train, val, test = [Batchifyer(*dataset, 100, 1) for dataset in ml.split(images, bboxes, weights=[0.7, 0.2, 0.1])]

for images, bboxes in test:
    for image, bboxe in zip(images, bboxes):
        f, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0))
        ml.plot_bounding_boxes(bboxe, ax)
        plt.show()