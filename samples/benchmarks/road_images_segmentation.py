import pathlib
import IPython
import numpy as np
import pygmalion as ml
import pygmalion.neural_networks as nn
import matplotlib.pyplot as plt
import math
import torch
import torch.nn.functional as F
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# Download the data
ml.datasets.cityscapes(data_path)

# Load data
data = dict(np.load(data_path / "cityscapes.npz"))
classes = [c.decode("utf-8") for c in data["classes"].tolist()]
fractions = data["fractions"]
colors = data["colors"]
# class_weights = (1/fractions)**0.5
x_train = data["train_images"]
y_train = data["train_segmented"]
x_test = data["test_images"]
y_test = data["test_segmented"]

# Create and train the model
device = "cuda:0"
model = nn.ImageSegmenter(3, classes, [16, 32, 64, 128, 256, 512], pooling_size=(2, 2), kernel_size=(3, 3),
                          n_convs_per_block=2, dropout=0.5, low_memory=True, upsampling_method="nearest", entropy_dice_mixture=1.0)
model.to(device)

class Batchifyer:

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 n_batches: int=1, batch_size: int=100,
                 device: torch.device = torch.device("cpu"),
                 data_augmentation: bool = False,
                 max_rotation_angle: float = 5.0,
                 max_downscaling_factor: float = 0.95):
        self.x, self.y = model.data_to_tensor(x, y)
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.device = device
        self.data_augmentation = data_augmentation
        self.max_rotation_angle = max_rotation_angle
        self.max_downscaling_factor = max_downscaling_factor

    def __iter__(self):
        shuffle = np.random.permutation(len(self.x))
        _, _, h, w = self.x.shape
        for i in range(self.n_batches):
            idx = shuffle[i*self.batch_size:(i+1)*self.batch_size]
            X, Y = self.x.to(self.device)[idx], self.y.to(self.device)[idx]
            if self.data_augmentation:
                n = len(idx)
                max_theta = self.max_rotation_angle * (math.pi / 180.0)
                wh = torch.tensor([w, h], dtype=torch.float32, device=self.device).reshape(1, 1, 1, 2)
                grid = torch.stack(torch.meshgrid([torch.linspace(-w/2, w/2, w, device=self.device),
                                                   torch.linspace(-h/2, h/2, h, device=self.device)],
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
                Y = F.grid_sample(Y.unsqueeze(1).float(), grid, mode="nearest", align_corners=True, padding_mode="reflection").squeeze(1).long()
            yield X, Y

(x_val, y_val), (x_test, y_test) = ml.split(x_test, y_test, weights=[400, 100])
train_data = Batchifyer(x_train, y_train, batch_size=50, n_batches=1, data_augmentation=True)
val_data = Batchifyer(x_val, y_val, batch_size=100, n_batches=1)
train_losses, val_losses, grad_norms, best_step = model.fit(train_data, val_data,
    n_steps=15000, learning_rate=1.0E-4, patience=None, keep_best=True, L2=0.3)

# Plot results
ml.plot_losses(train_losses, val_losses, grad_norms, best_step)
y_predicted = model.predict(x_test[:10])
for x, y_t, y_p in zip(x_test[:10], y_test[:10], y_predicted[:10]):
    f, axes = plt.subplots(figsize=[15, 5], ncols=3)
    for im, ax, title in zip([x, y_p, y_t], axes,
                             ["image", "segmented", "target"]):
        if len(im.shape) == 2:
            h, w = im.shape
            n, c = colors.shape
            im = np.take_along_axis(colors.reshape(1, 1, n, c),
                                    im.reshape(h, w, 1, 1),
                                    axis=-2).reshape(h, w, c)
        ax.imshow(im)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    f.tight_layout()
plt.show()

IPython.embed()
