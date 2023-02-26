import IPython
import numpy as np
import pygmalion as ml
import pygmalion.neural_networks as nn
import matplotlib.pyplot as plt
plt.style.use("bmh")
DEVICE = "cuda:0"

# Download the data
train_data = ml.datasets.generators.CirclesGenerator(100, 1, device=DEVICE)

# Create and train the model
model = nn.ImageSegmenter(1, ["inside", "outside"], [8, 16, 32, 64], pooling_size=(2, 2), n_convs_per_block=2)
model.to(DEVICE)
train_losses, val_losses, best_step = model.fit(train_data,
    n_steps=5000, learning_rate=1.0E-3, patience=None, keep_best=False)

# Plot results
ml.plot_losses(train_losses, val_losses, best_step)
x_train, y_train = train_data.generate(5)
y_predicted = model.predict(x_train)
y_train, y_predicted = (v.astype(np.uint8) * 255 for v in (y_train, y_predicted))
for x, y_t, y_p in zip(x_train, y_train, y_predicted):
    f, axes = plt.subplots(figsize=[15, 6], ncols=3)
    for im, ax, title in zip([x, y_p, y_t], axes,
                             ["image", "segmented", "target"]):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    f.tight_layout()
plt.show()

IPython.embed()
