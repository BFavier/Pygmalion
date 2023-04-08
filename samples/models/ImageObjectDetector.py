import pathlib
import numpy as np
import pygmalion as ml
import matplotlib.pyplot as plt
plt.style.use("bmh")
data_path = pathlib.Path(__file__).parents[1] / "data"

# # Download the data
# ml.datasets.aquarium(data_path)

# data = dict(np.load(data_path / "aquarium.npz"))
# images = data["images"]
# image_indexes = data["image_indexes"]
# classes = [c.decode("utf-8") for c in data["class_names"].tolist()]
# bboxes = [{"class": [classes[i] for i in data["class_indexes"][image_indexes == i]],
#             **{k: data[k][image_indexes == i] for k in ("x", "y", "w", "h")}}
#            for i in range(len(images))]

# for i in np.random.permutation(len(images)):
#     f, ax = plt.subplots()
#     ax.imshow(images[i])
#     ml.plot_bounding_boxes(bboxes[i], ax)
#     plt.show()

generator = ml.datasets.generators.ShapesGenerator(10, 1)

images, bboxes = next(iter(generator))

for image, bboxe in zip(images, bboxes):
    f, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ml.plot_bounding_boxes(bboxe, ax, class_colors={"circle": "r", "square": "b"})
    plt.show()

# if __name__ == "__main__":
#     import IPython
#     IPython.embed()
