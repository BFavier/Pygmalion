import torch
import pygmalion as ml
import matplotlib.pyplot as plt

DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
model = ml.neural_networks.ImageObjectDetector(1, ["circle", "square"],
                                               features=[8, 16, 32, 64],
                                               bboxes_per_cell=5, kernel_size=(3, 3),
                                               pooling_size=(2, 2), n_convs_per_block=2)
model.to(DEVICE)

class Batchifyer:

    def __init__(self, batch_size: int, n_batches: int):
        self.generator = ml.datasets.generators.ShapesGenerator(batch_size, n_batches)

    def __iter__(self):
        for x, y in self.generator:
            yield model.data_to_tensor(x, y)


data = Batchifyer(100, 1)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98))
train_loss, val_loss, grad, best_step = model.fit(training_data=data, optimizer=optimizer, n_steps=1000, keep_best=False, learning_rate=lambda step: 1.0E-3/(0.1 * step**0.5 + 1))
ml.utilities.plot_losses(train_loss, val_loss, grad, best_step)
plt.show()

x, y_target = data.generator.generate(10)
y_pred = model.predict(x, detection_treshold=0.5, threshold_intersect=0.8)
h_grid, w_grid = model.cells_dimensions
for img, bboxes, bboxes_pred in zip(x, y_target, y_pred):
    h, w = img.shape[:2]
    f, (ax1, ax2) = plt.subplots(figsize=[10, 5], ncols=2)
    ax1.set_title("predicted")
    ax1.imshow(img, cmap="gray")
    for i in range(0, h-1, h_grid):
        ax1.axvline(i-0.5, color="#00ff00")
    for i in range(0, w-1, w_grid):
        ax1.axhline(i-0.5, color="#00ff00")
    ax1.scatter(bboxes_pred["x"], bboxes_pred["y"], c=["b" if c == "square" else "r" for c in bboxes_pred["class"]],
                marker="x")
    ax1.set_xlim([0, w-1])
    ax1.set_ylim([h-1, 0])
    ml.utilities.plot_bounding_boxes(bboxes_pred, ax1, class_colors={"circle": "r", "square": "b"})
    ax2.set_title("target")
    ax2.imshow(img, cmap="gray")
    ml.utilities.plot_bounding_boxes(bboxes, ax2, class_colors={"circle": "r", "square": "b"})
plt.show()


if __name__ == "__main__":
    import IPython
    IPython.embed()
