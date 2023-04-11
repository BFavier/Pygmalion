import torch
import numpy as np
import pygmalion as ml
import matplotlib.pyplot as plt

model = ml.neural_networks.ImageObjectDetector(1, ["circle", "square"],
                                               features=[8, 16, 32, 64],
                                               bboxes_per_cell=5, kernel_size=(3, 3),
                                               pooling_size=(2, 2), n_convs_per_block=2)
model.to("cuda:0")

class Batchifyer:

    def __init__(self, batch_size: int, n_batches: int):
        self.generator = ml.datasets.generators.ShapesGenerator(batch_size, n_batches)

    def __iter__(self):
        for x, y in self.generator:
            yield model.data_to_tensor(x, y)


data = Batchifyer(100, 1)
optimizer = torch.optim.Adam(model.parameters())
train_loss, val_loss, grad, best_step = model.fit(training_data=data, optimizer=optimizer, n_steps=1000, keep_best=False, learning_rate=1.0E-3)
ml.plot_losses(train_loss, val_loss, grad, best_step)
plt.show()

x, y_target = data.generator.generate(10)
y_pred = model.predict(x, detection_treshold=0.5, threshold_intersect=0.8)
for img, bboxes, bboxes_pred in zip(x, y_target, y_pred):
    f, (ax1, ax2) = plt.subplots(figsize=[10, 5], ncols=2)
    ax1.set_title("predicted")
    ax1.imshow(img, cmap="gray")
    ml.plot_bounding_boxes(bboxes_pred, ax1, class_colors={"circle": "r", "square": "b"})
    ax2.set_title("target")
    ax2.imshow(img, cmap="gray")
    ml.plot_bounding_boxes(bboxes, ax2, class_colors={"circle": "r", "square": "b"})
plt.show()


if __name__ == "__main__":
    import IPython
    IPython.embed()
