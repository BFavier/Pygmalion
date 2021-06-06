Pygmalion in the greek mythologie is a sculptor that fell in love with one of his creations.
In the myth, Aphrodite gives life to Galatea, the sculpture he fell in love with. This package is a machine learning library. It contains all the tools you need to give a mind of their own to inanimate objects.

# Installing pygmalion

pygmalion can be installed through pip.

~~~
pip install pygmalion
~~~

# Fast prototyping of models with pygmalion

Architectures for several common machine learning tasks (regression, image classification, ...) are implemented in this package.

The inputs and outputs of the models are common python objects (such as numpy array and pandas dataframes) so there are few new things you need to learn to use this package.

In this part we are going to see how to load a dataset, train a model, and display some metrics. As a first step you can import the following packages.

~~~python
>>> import pygmalion as ml
>>> import pygmalion.neural_networks as nn
>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt
~~~

You can download a dataset and split it with the **split** function.

~~~python
>>> ml.datasets.boston_housing("./")
>>> df = pd.read_csv("./boston_housing.csv")
>>> x, y = df[[c for c in d_Fcolumns if c != "medv"]], df["medv"]
>>> train_data, val_data, test_data = ml.split(x, y, frac=(0.1, 0.1))
~~~

Creating and training a model is done in a few lines of code.

~~~python
>>> hidden_layers = [{"features": 8}, {"features": 8}]
>>> model = nn.DenseRegressor(x.columns, hidden_layers)
>>> model.train(train_data, val_data, n_epochs=1000, patience=100, learning_rate=1.0E-3)
~~~

Some usefull metrics can easily be evaluated.

For a regressor model, the available metrics are [**MSE**](https://en.wikipedia.org/wiki/Mean_squared_error), [**RMSE**](https://en.wikipedia.org/wiki/Root-mean-square_deviation), [**R2**](https://en.wikipedia.org/wiki/Coefficient_of_determination), and the correlation between target and prediction can be visualized with the **plot_fitting** function.

~~~python
>>> f, ax = plt.subplots()
>>> x_train, y_train = train_data
>>> ml.plot_fitting(model(x_train), y_train, ax=ax, label="training")
>>> x_val, y_val = val_data
>>> ml.plot_fitting(model(x_val), y_val, ax=ax, label="validation")
>>> x_test, y_test = test_data
>>> ml.plot_fitting(model(x_test), y_test, ax=ax, label="testing", color="C3")
>>> R2 = ml.R2(model(x_test), y_test)
>>> ax.set_title(f"R²={R2:.3g}")
>>> plt.show()
~~~

![pairplot](https://raw.githubusercontent.com/BFavier/Pygmalion/main/images/boston_housing_pairplot.png)


For a classifier model you can evaluate the [**accuracy**](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification), and display the confusion matrix.

~~~python
>>> ml.datasets.iris("./")
>>> df = pd.read_csv("./iris.csv")
>>> x, y = df[[c for c in d_Fcolumns if c != "variety"]], df["variety"]
>>> inputs, classes = x.columns(), y.unique()
>>> hidden_layers = [{"features": 5},
>>>                  {"features": 5},
>>>                  {"features": 5}]
>>> model = nn.DenseClassifier(inputs, classes,
>>>                            hidden_layers=hidden_layers,
>>>                            activation="elu")
>>> train_data, val_data, test_data = ml.split(x, y, frac=(0.2, 0.1))
>>> model.train(train_data, val_data, n_epochs=1000, patience=100)
>>> f, ax = plt.subplots()
>>> x_test, y_test = test_data
>>> ml.plot_matrix(ml.confusion_matrix(y_test, y_pred, classes=classes),
...                ax=ax, cmap="Greens", write_values=True, format=".2%")
>>> acc = ml.accuracy(y_pred, y)*100
>>> ax.set_title(f"Accuracy: {acc:.2f}%")
>>> plt.tight_layout()
>>> plt.show()
~~~

![confusion matrix](https://raw.githubusercontent.com/BFavier/Pygmalion/main/images/iris_confusion_matrix.png)

All the models can be dumped as a dictionnary through the **dump** property. A copy of the model can be loaded with the **from_dump** class method.

~~~python
>>> dump = model.dump
>>> model = nn.DenseRegressor.from_dump(dump)
~~~

The models can also be be saved directly to the disk in json/hdf5 format with the **save** method.
A model saved on the disk can then be loaded back with the **load** function.

~~~python
>>> model.save("./model.json")
>>> model = ml.load("./model.json")
~~~

# Implemented models

For examples of model training see the **samples** folder in the [github page](https://github.com/BFavier/Pygmalion).

## Neural networks

The neural networks are implemented in pytorch under the hood.
The underlying pytorch Module and Optimizer can be accessed as the **model** and **optimizer** attributes of the model.

### **DenseRegressor**

A dense regressor (or multi layer perceptron regressor) predicts a scalar value given an input of several variables.

This implementation takes in input **x** a pandas.DataFrame of numerical observations, and returns **y** a numpy.ndarray of floats of the same length. The optional **weights** weighting of the observations during training are numpy.ndarray of floats.

It is implemented as a sucession of hidden **Activated0d** layers (linear weighting/non linear activation/batch normalization) and a final linear weighting to reduces the number of features to one scalar prediction.

### **DenseClassifier**

A dense classifier (or multi layer perceptron classifier) predicts a str class value given an input of several variables.

This implementation takes in input **x** a pandas.DataFrame of numerical observations, and returns **y** list of str of the same length. The optional **weights** weighting of the observations during training are numpy.ndarray of floats.

Similarly to the DenseRegressor it is a succession of hidden **Activated0d** layers, and a final linear layer with as much output as there are classes to predict.

### **ImageClassifier**

An ImageClassifier predicts a str class given as input an image. Here below the predictions of a model trained on the fashion-MNIST dataset.

![fashion-MNIST predictions](https://raw.githubusercontent.com/BFavier/Pygmalion/main/images/Fashion_MNIST_illustration.png)

It is implemented as a Convolutional Neural Network similar to LeNet.

### **SemanticSegmenter**

A SemanticSegmenter predicts a class for each pixel of the input image. Here below the predictions of a model trained on the cityscape dataset.

![segmented_cityscapes](https://raw.githubusercontent.com/BFavier/Pygmalion/main/images/segmented_cityscape_2.png)

It is implemented as a Convolutional Neural Network similar to U-Net. It is a succession of convolutions/pooling followed by a succession of upsampling/convolutions, leading to a convergent/divergent feature map structure. The feature map before each downsampling stage is concatenated to the upsampling of the same size to preserve features.


