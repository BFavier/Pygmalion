# Pygmalion

Pygmalion in the greek mythologie is a sculptor that fell in love with one of his creations.
In the story, Aphrodite gives life to the sculpture.
Similarly, the purpose of this package is to help you give a mind of their own to inanimate objects.

## Fast prototyping of tried-and-tested model types

Some common model architectures are implemented in this package for several common machine learning tasks.
Creating, training, and evaluating performances of a model requires few lines of code.
The inputs and outputs of the model are common python object (such as numpy array and pandas dataframes)

Here below a piece of code to train a multi-layer perceptron regressor on the Boston housing dataset.

~~~python
>>> import pygmalion as ml
>>> import pygmalion.neural_networks as nn
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> # Download the dataset in the current folder
>>> ml.datasets.boston_housing("./")
>>> # Load the dataset
>>> df = pd.read_csv("boston_housing.csv")
>>> x, y = df[[c for c in df.columns if c != "medv"]], df["medv"]
>>> # split the data
>>> data, test_data = ml.split((x, y), frac=0.2)
>>> train_data, val_data = ml.split(data, frac=0.2)
>>> # Create the model
>>> hidden_layers = [{"channels": 16}, {"channels": 8}, {"channels": 4}]
>>> model = nn.DenseRegressor(len(x.columns), hidden_layers, learning_rate=1.0E-3, GPU=False)
>>> # train the model
>>> model.train(train_data, val_data, n_epochs=1000, patience=100)
>>> # Evaluate some metrics
>>> x_train, y_train = train_data
>>> x_val, y_val = val_data
>>> x_test, y_test = test_data
>>> f, ax = plt.subplots()
>>> model.plot_correlation(model(x_train), y_train, ax=ax, label="training")
>>> model.plot_correlation(model(x_val), y_val, ax=ax, label="validation")
>>> ax.set_title(f"R²={ml.R2(model(x_test), y_test)}")
>>> plt.show()
~~~

## Model loading and saving

All the model types can be easily be saved to the disk with the "save" method.
A model saved on the disk can be loaded back with the "load" class method.

~~~python
>>> import numpy as np
>>> model.save("./model.json")
>>> y1 = model(x)
>>> model = nn.DenseRegressor.load("./model.json")
>>> y2 = model(x)
>>> print(np.allclose(y1 - y2))
~~~

## Implemented models

### Neural networks

The neural networks available in this package are implemented using pytorch, and training on GPU is possible. Most of the complexity is hidden to you (training loop, )

