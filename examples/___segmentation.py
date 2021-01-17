import matplotlib.pyplot as plt
import machine_learning as ml
import skimage.data as data
import skimage.filters as filt
import numpy as np

names = ["astronaut", "camera", "checkerboard", "immunohistochemistry",
         "chelsea", "coins", "hubble_deep_field", "rocket",
         "page", "retina", "coffee"]

images = []
for name in names:
    func = getattr(data, name)
    array = func()
    image = ml.Image(data=array).as_gray().resized(100, 100)
    images.append(image)


def segment(im):
    threshold = filt.threshold_sauvola(im.data, window_size=5)
    mask = im.data > threshold
    data = np.zeros([im.height, im.width, 3], dtype=np.uint8)
    data[~mask] = [255, 0, 0]
    data[mask] = [0, 255, 0]
    return ml.Image(data=data)


segmented = [segment(im) for im in images]

train = images, segmented
val = [im.flipped(x=True) for im in images], [seg.flipped(x=True)
                                              for seg in segmented]

model = ml.neural_networks.ImageSegmenter()
model.fit(*train, validation=val, learning_rate=1.0E-3,
          down_windows=[(5, 5), (4, 4), (2, 2)],
          down_channels=[16, 32, 64],
          pooling_windows=[(2, 2), (2, 2), (2, 2)],
          up_windows=[(2, 2), (4, 4), (6, 6)],
          up_channels=[16, 8, 3],
          non_linear="relu", n_epochs=1000, patience=30)

model.plot_history(log=True)
for i, im in enumerate(images):
    se = segmented[i]
    pr = model.predict(im)
    f, axes = plt.subplots(nrows=1, ncols=3, figsize=[15, 5])
    im.draw(ax=axes[0])
    axes[0].set_title("original")
    se.draw(ax=axes[1])
    axes[1].set_title("segmented")
    pr.draw(ax=axes[2])
    axes[2].set_title("predicted")
    plt.suptitle(names[i])
plt.show()
