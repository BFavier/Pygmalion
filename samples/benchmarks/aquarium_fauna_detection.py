import pathlib
import torch
import torch.nn.functional as F
import numpy as np
import pygmalion as ml
import matplotlib.pyplot as plt
from typing import Optional
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

model = ml.neural_networks.ImageObjectDetector(3, classes, [8, 16, 32, 64, 128], n_convs_per_block=2, dropout=0.1)
model.to("cuda:0")

class Batchifyer:

    def __init__(self, images: np.ndarray, bboxes: list, batch_size: int, n_batches: int,
                 data_augmentation: bool = True, device: torch.device = "cpu",
                 max_rotation_angle: float = 5.0, multi_scale: bool = True,
                 resolution_range: Optional[tuple[int, int]] = (32, 64)):
        self.images, self.bboxes = model._x_to_tensor(images), bboxes
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.device = device
        self.data_augmentation = data_augmentation
        self.max_rotation_angle = max_rotation_angle
        self.resolution_range = resolution_range
        self.multiscale = multi_scale

    def __iter__(self):
        shuffle = np.random.permutation(len(self.images))
        _, _, image_h, image_w = self.images.shape
        cell_h, cell_w = model.cells_dimensions
        for i in range(self.n_batches):
            idx = shuffle[i*self.batch_size:(i+1)*self.batch_size]
            X, Y = self.images.to(self.device)[idx], [self.bboxes[i] for i in idx]
            while (image_h // cell_h) > 0 and (image_w // cell_w) > 0:
                if self.data_augmentation:
                    n = len(idx)
                    max_theta = self.max_rotation_angle * (math.pi / 180.0)
                    wh = torch.tensor([image_w, image_h], dtype=torch.float32, device=self.device).reshape(1, 1, 1, 2)
                    grid = torch.stack(torch.meshgrid([torch.linspace(-image_w/2, image_w/2, image_w, device=self.device),
                                                       torch.linspace(-image_h/2, image_h/2, image_h, device=self.device)],
                                    indexing="xy"), dim=-1)
                    # random horizontal flips
                    one = torch.ones(n, device=self.device)
                    factor = torch.stack([torch.where(torch.rand(n, device=self.device) > 0.5, one, -one), one], dim=-1)
                    grid = grid.reshape(1, image_h, image_w, 2) * factor.reshape(n, 1, 1, 2)
                    # random rotations
                    theta = torch.rand(n, device=self.device) * (2*max_theta) -  max_theta
                    sin, cos = torch.sin(theta), torch.cos(theta)
                    rot = torch.stack([torch.stack([cos, -sin], dim=-1),
                                       torch.stack([sin, cos], dim=-1)], dim=-2)
                    grid = (rot.reshape(n, 1, 1, 2, 2) @ grid.reshape(n, image_h, image_w, 2, 1)).squeeze(-1)
                    # scaling to stay in image range
                    grid = grid / (grid*2/wh).reshape(n, -1).abs().max(dim=1).values.reshape(n, 1, 1, 1)
                    # random offset
                    delta = (wh.reshape(1, 2)/2 - grid.reshape(n, -1, 2).max(dim=1).values)
                    offset = torch.rand((n, 2), device=self.device) * (2*delta) - delta
                    grid = grid + offset.reshape(n, 1, 1, 2)
                    # converting back to (-1, 1) range
                    grid = torch.clip(grid*2 / wh, -1, 1)
                    Xinterp = F.grid_sample(X, grid, mode="bilinear", align_corners=True, padding_mode="border")
                    # converting bboxe coordinates
                    wh = wh.reshape(1, 2)
                    coords = [tuple(Yb[k] for k in "xywh") for Yb in Y]
                    corners = [torch.stack([torch.stack([torch.tensor(x, dtype=torch.float) - torch.tensor(w, dtype=torch.float)/2,
                                                         torch.tensor(y, dtype=torch.float) - torch.tensor(h, dtype=torch.float)/2], dim=-1)/wh,
                                            torch.stack([torch.tensor(x, dtype=torch.float) + torch.tensor(w, dtype=torch.float)/2,
                                                         torch.tensor(y, dtype=torch.float) - torch.tensor(h, dtype=torch.float)/2], dim=-1)/wh,
                                            torch.stack([torch.tensor(x, dtype=torch.float) + torch.tensor(w, dtype=torch.float)/2,
                                                         torch.tensor(y, dtype=torch.float) + torch.tensor(h, dtype=torch.float)/2], dim=-1)/wh,
                                            torch.stack([torch.tensor(x, dtype=torch.float) - torch.tensor(w, dtype=torch.float)/2,
                                                         torch.tensor(y, dtype=torch.float) + torch.tensor(h, dtype=torch.float)/2], dim=-1)/wh], dim=1)
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
                    # create interpolated bboxes
                    Yinterp = [{"class": cls, "x": _x, "y": _y, "w": _w, "h": _h}
                               for _x, _y, _w, _h, cls in zip(x, y, w, h, classes)]
                else:
                    Xinterp, Yinterp = X, Y
                if self.multiscale:
                    # filter based on bboxe size
                    if self.resolution_range is not None:
                        inf, sup = min(self.resolution_range), max(self.resolution_range)
                        x, y, w, h, classes = ([Yb[k] for Yb in Yinterp] for k in ("x", "y", "w", "h", "class"))
                        length = [np.minimum(_w, _h) for _w, _h in zip(w, h)]
                        keep = [(_l >= inf) & (_l <= sup) for _l in length]
                        x, y, w, h = ([b[k].tolist() for b, k in zip(v, keep)] for v in (x, y, w, h))
                        classes = [[c for c, k in zip(cls, kp) if k] for cls, kp in zip(classes, keep)]
                        Yinterp = [{"class": cls, "x": _x, "y": _y, "w": _w, "h": _h}
                                for _x, _y, _w, _h, cls in zip(x, y, w, h, classes)]
                yield (Xinterp, *model._y_to_tensor(Yinterp, image_h, image_w, device=self.device))
                # update image dimensions or exit
                if self.multiscale:
                    h_down, w_down = model.downsampling_window
                    X = F.avg_pool2d(X, model.downsampling_window)
                    image_h, image_w = X.shape[-2:]
                    Y = [{"class": Yb["class"],
                          "x": [x / w_down for x in Yb["x"]],
                          "y": [y / h_down for y in Yb["y"]],
                          "w": [w / w_down for w in Yb["w"]],
                          "h": [h / h_down for h in Yb["h"]]}
                         for Yb in Y]
                else:
                    break

train, val, (x_test, y_test) = ml.split(images, bboxes, weights=[0.8, 0.1, 0.1])
train = Batchifyer(*train, 40, 1, data_augmentation=True, multi_scale=True)
val = Batchifyer(*val, 40, 1, data_augmentation=True, multi_scale=True)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98))
train_loss, val_loss, grad, best_step = model.fit(training_data=train, validation_data=val, optimizer=optimizer,
                                                  n_steps=10000, keep_best=False,
                                                  learning_rate=1.0E-3)
ml.plot_losses(train_loss, val_loss, grad, best_step)
plt.show()

y_pred = model.predict(x_test, detection_treshold=0.5, threshold_intersect=0.8, multi_scale=True)
for img, bboxes, bboxes_pred, _ in zip(x_test, y_test, y_pred, range(10)):
    h, w = img.shape[:2]
    f, (ax1, ax2) = plt.subplots(figsize=[10, 5], ncols=2)
    ax1.set_title("predicted")
    ax1.imshow(img, cmap="gray")
    ax1.set_xlim([0, w-1])
    ax1.set_ylim([h-1, 0])
    ml.plot_bounding_boxes(bboxes_pred, ax1, class_colors={"circle": "r", "square": "b"})
    ax2.set_title("target")
    ax2.imshow(img, cmap="gray")
    ml.plot_bounding_boxes(bboxes, ax2, class_colors={"circle": "r", "square": "b"})
plt.show()

if __name__ == "__main__":
    import IPython
    IPython.embed()
